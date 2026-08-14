#!/usr/bin/env python
'''
XViewUNet: per-view UNet trunks followed by time-banded cross-view self-attention.

Motivation: DNNROI-style ROI finding from deconvolved images only, with no
MP2/MP3 coincidence-mask input channels.  Each detector view (e.g. U, V, W) is
processed by its own UNet trunk.  The trunk feature maps are chunked along the
electronics-channel axis into tokens and all views' tokens within a small time
window attend to each other ("banded" self-attention: every token attends to
all channels of all views at ticks within +/- band).  The attention output is
fused back with the full-resolution trunk features to produce per-pixel logits.

The attention path is gated by zero-initialized scalars (LayerScale/ReZero
style) so a freshly constructed (or trunk-warm-started) model produces exactly
the per-view UNet outputs.  Training can therefore start from pretrained
per-view UNet checkpoints (see unet_checkpoints) with trunks frozen
(freeze_unets) and only later be fine-tuned end to end (init_checkpoint).

Two training settings are load-bearing, both consequences of that zero-init
gating:

- Autocast in bfloat16, not float16.  Once the gates open, the cross-view
  branch overflows in fp16 and emits inf logits for a large fraction of
  batches (~29% as measured).  GradScaler then skips those steps, so the
  visible symptom is a whole epoch of nan loss while the weights stay finite
  and those optimizer steps are silently discarded.
- Use an Adam-family optimizer, not SGD.  While gamma is zero the gradient
  reaching the branch weights is exactly zero and gamma's own gradient is
  tiny, so an SGD step -- being gradient-proportional -- never opens the gates
  and the loss sits at exactly the frozen-trunk baseline.  Adam normalises per
  parameter and escapes.  Note the lr that suits SGD here is far too large for
  Adam.

The gating also makes this model get *riskier* as it trains, which is the
opposite of the usual assumption.  While gamma is near zero the gradient
reaching the branch weights is proportional to it, so early training is
intrinsically stable; as the gates open that gradient grows, and a rate that
was safe at gamma=0 can diverge at gamma=0.3.  Observed at lr=4e-3: gammas
opened smoothly for six epochs (to ~0.29, 0.27, -0.37) and then collapsed in
one epoch (to ~0.05, -0.09, -0.12, one of them through zero), reverting the
output toward the plain trunk baseline and undoing the gain.  The loss recovers
only partially afterwards and in a different configuration, so a restart from
before the collapse beats waiting.  Keep the rate at or below the 1e-3 the
optimizer's own guard implies, and watch the gamma magnitudes rather than the
loss alone -- the loss moves a beat later than the gates do.

Because the gates start closed, a fresh model is bit-exact equal to its trunks
run per view and segment, so the frozen-trunk baseline is the number to beat.
Compute it before training rather than reading it off epoch 0, since the
optimizer perturbs the model from the first step.

This model cannot be TorchScript-*scripted*, though the other apps can be: the
runtime-int indexing of nn.ModuleList, the callable-plus-args signature of
_ckpt and the tuple-keyed _mask_cache are all compile-time blockers, so eval()
mode does not avoid them.  Export by tracing instead ("wcpy dnn export-ts -m
trace"), which is bit-exact.

References for the zero-init residual gating:

- ReZero: Bachlechner et al., "ReZero is All You Need: Fast Convergence at
  Large Depth", arXiv:2003.04887 (2020).  One zero-init scalar per residual
  branch, to train deep stacks without warmup.
- LayerScale: Touvron et al., "Going deeper with Image Transformers"
  (CaiT), arXiv:2103.17239 (2021).  Per-channel diagonal vector, initialized
  to a small constant rather than to exactly zero.
- SkipInit: De & Smith, "Batch Normalization Biases Residual Blocks Towards
  the Identity Function", arXiv:2002.10444 (2020).

Those papers reach for the technique to stabilize depth.  The gammas here take
LayerScale's per-channel shape but ReZero's exactly-zero value, for a different
end: exact zero makes the branch vanish identically, which is what buys the
bit-exact trunk equivalence above.  It is also precisely what stalls SGD --
LayerScale's small-but-nonzero init would avoid that, at the cost of the
bit-exactness.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .unet import UNet, dconv

import logging
log = logging.getLogger("wirecell.dnn")


#: Attention scopes, selectable at runtime for ablation.  A query token may see
#: keys from:
#:
#:   legacy  every token, i.e. the behaviour before modes existed.  Kept so a
#:           checkpoint trained under it stays reproducible; it is the only
#:           mode that lets the two W faces attend to each other.
#:   all     its own segment, or any segment of a different view.
#:   intra   its own segment only.
#:   inter   segments of a different view only.
#:   none    nothing; the attention branch is skipped entirely, though the
#:           block's FFN branch still runs.
#:
#: 'intra' and 'inter' partition 'all' exactly: every ordered token pair is
#: either same-segment or different-view, apart from the same-view-different-
#: segment pairs that 'all' excludes and 'legacy' allows.
ATTN_MODES = ('legacy', 'all', 'intra', 'inter', 'none')


def _gather(t, spans):
    '''
    Token-axis slice of t (B, T, N, d) over [(lo, hi), ...].  Out of place, so
    it stays safe under activation checkpointing; a single span avoids the copy.
    '''
    if len(spans) == 1:
        lo, hi = spans[0]
        return t[:, :, lo:hi, :]
    return torch.cat([t[:, :, lo:hi, :] for lo, hi in spans], dim=2)


class BandedAttentionBlock(nn.Module):
    '''
    Pre-LN transformer block with time-banded self-attention.

    Input/output tokens are shaped (B, T, N, d): at each tick t, the N tokens
    (all views' channel chunks) attend to all N tokens at every tick within
    t +/- band.  Implemented with rolled key/value copies so a single batched
    scaled_dot_product_attention call covers all ticks; rolled-in wraparound
    keys at the time edges are masked off.

    Both residual branches are gated by zero-initialized LayerScale vectors so
    the block is the identity at construction.
    '''

    def __init__(self, d_model, n_heads, band=1, ffn_mult=4):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f'd_model={d_model} not divisible by n_heads={n_heads}')
        self.d_model = d_model
        self.n_heads = n_heads
        self.band = int(band)

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3*d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.gamma1 = nn.Parameter(torch.zeros(d_model))

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult*d_model),
            nn.GELU(),
            nn.Linear(ffn_mult*d_model, d_model),
        )
        self.gamma2 = nn.Parameter(torch.zeros(d_model))

        self._mask_cache = dict()  # (T, N) -> (T, (2*band+1)*N) bool, True = attend

        # Attention scope.  Plain attributes, not buffers or parameters, so
        # switching mode never touches state_dict and one trained checkpoint
        # can be evaluated under every mode.  segments is filled in by
        # XViewUNet; a block built standalone stays in 'legacy'.
        self.segments = None       # [(lo, hi, view), ...] in token order
        self.attn_mode = 'legacy'

    def _band_mask(self, T, N, device):
        '''
        Boolean (T, (2*band+1)*N) mask, True where a query at tick t may attend
        to a key column.  The key axis is the band offsets concatenated in
        s = -band..+band order, N columns each, matching how forward() builds
        its rolled key/value copies.

        Validity is a function of time alone: it is computed per offset from
        whether t+s lands inside [0, T), then broadcast unchanged across all N
        tokens.  So the N token columns of a given offset are always all-True
        or all-False together -- nothing here distinguishes one channel chunk,
        segment or view from another.  That is the intended design: within the
        band, attention over the token axis is dense, every view's channel
        chunks seeing every other's.  Any per-channel or per-view restriction
        (a geometry or wire-line bias, say) would have to be added separately.

        The only entries masked off are therefore the wraparound ones: the
        rolls in forward() are circular, so near t=0 and t=T-1 some rolled
        columns hold keys from the opposite end of the image, which are not
        real neighbours in time.

        Cached per (T, N) since the mask depends only on shape.
        '''
        key = (T, N)
        mask = self._mask_cache.get(key)
        if mask is None or mask.device != device:
            t = torch.arange(T)
            cols = list()
            for s in range(-self.band, self.band + 1):
                valid = ((t + s) >= 0) & ((t + s) < T)   # (T,)
                cols.append(valid.unsqueeze(1).expand(T, N))
            mask = torch.cat(cols, dim=1).to(device)     # (T, (2b+1)*N)
            self._mask_cache[key] = mask
        return mask

    def forward(self, x):
        '''
        Tokens (B, T, N, d) in, same shape out.

        B is the batch, T the number of time ticks (carried at full image
        resolution, one attention position per tick).  N is the total token
        count at a single tick: the trunk features of every (view, segment)
        have been chunked along the electronics-channel axis, giving
        segment_width // chunk tokens per segment, and all of those are
        concatenated across all views and segments into one flat axis.  So a
        token is "one channel chunk of one segment of one view", and the N
        axis mixes views deliberately -- that is what makes the attention
        cross-view.  d is d_model, the per-token embedding width.

        The band is covered without a loop over ticks: rolling the keys and
        values by -s puts tick t+s at position t, so concatenating the 2*band+1
        rolls along the token axis gives every query its whole neighbourhood in
        one batched attention call, at the cost of materialising that many
        copies of k and v.
        '''
        if self.attn_mode == 'none':
            # Attention branch off, FFN branch still on, so an ablation against
            # another mode isolates attention rather than also removing the
            # block's extra capacity.
            return x + self.gamma2 * self.ffn(self.norm2(x))

        y = self.norm1(x)

        '''Create the queries, keys, values used with scaled dot product attn.'''
        q, k, v = self.qkv(y).chunk(3, dim=-1)

        if self.attn_mode == 'legacy':
            o = self._attend(q, k, v)
        else:
            if not self.segments:
                raise RuntimeError(
                    f'attn_mode={self.attn_mode!r} needs segment layout; this '
                    'block was built standalone rather than by XViewUNet')
            outs = list()
            for seg in self.segments:
                lo, hi, _ = seg
                qs = q[:, :, lo:hi, :]
                spans = self._key_spans(seg)
                outs.append(self._attend(qs, _gather(k, spans), _gather(v, spans))
                            if spans else torch.zeros_like(qs))
            o = torch.cat(outs, dim=2)

        x = x + self.gamma1 * self.proj(o)
        x = x + self.gamma2 * self.ffn(self.norm2(x))
        return x

    def _key_spans(self, seg):
        '''
        Token spans this query segment may attend to, merged so adjacent spans
        become one slice.  Under 'all' most segments end up with a single
        contiguous span, so the restriction usually costs no extra concatenation.
        '''
        lo, hi, view = seg
        keep = list()
        for (a, b, w) in self.segments:
            own = (a, b) == (lo, hi)
            if self.attn_mode == 'intra':
                ok = own
            elif self.attn_mode == 'inter':
                ok = w != view
            else:                                   # 'all'
                ok = own or w != view
            if ok:
                keep.append((a, b))
        merged = list()
        for a, b in sorted(keep):
            if merged and merged[-1][1] == a:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        return merged

    def _attend(self, q, k, v):
        '''
        Banded attention of q (B, T, Nq, d) against k/v (B, T, Nk, d).

        Nq and Nk need not match.  The modes restrict what a query may see by
        handing in a smaller key set rather than by masking a full grid: a
        per-query mask would need (B*T, 1, Nq, M), which at production size is
        ~10^9 entries, whereas a restricted key set keeps the mask at
        (B*T, 1, 1, M) and costs strictly fewer logits than 'legacy'.
        '''
        B, T, Nq, d = q.shape
        Nk = k.shape[2]
        h = self.n_heads
        dh = d // h

        '''
        Replicate keys/values from ticks t-band .. t+band: roll(k, -s)[t] == k[t+s]
        '''
        shifts = range(-self.band, self.band + 1)
        kb = torch.cat([torch.roll(k, -s, dims=1) for s in shifts], dim=2)
        vb = torch.cat([torch.roll(v, -s, dims=1) for s in shifts], dim=2)
        M = kb.shape[2]

        q = q.reshape(B*T, Nq, h, dh).transpose(1, 2)    # (B*T, h, Nq, dh)
        kb = kb.reshape(B*T, M, h, dh).transpose(1, 2)   # (B*T, h, M, dh)
        vb = vb.reshape(B*T, M, h, dh).transpose(1, 2)

        mask = self._band_mask(T, Nk, q.device)          # (T, M)
        mask = mask.repeat(B, 1).view(B*T, 1, 1, M)

        o = F.scaled_dot_product_attention(q, kb, vb, attn_mask=mask)
        return o.transpose(1, 2).reshape(B, T, Nq, d)


class XViewUNet(nn.Module):
    '''
    Per-view UNet trunks + banded cross-view attention + fused per-pixel head.

    Input is a single tensor (B, n_input_channels, sum-of-all-channels, T) with
    the views concatenated along the electronics-channel axis in view_splits
    order.  Output has the same shape with n_classes image channels (logits;
    no sigmoid -- pair with BCEWithLogitsLoss).

    - view_splits: per-view list of segment widths along the channel axis,
      e.g. [[800],[800],[480,480]] for PDHD U, V and the two W faces.  One
      trunk per view; a view's trunk runs once per segment so convolutions
      never straddle a face seam.  All segments of a view must have equal
      width (the trunk's umerge nodes memoize pad shapes).
    - chunks: per-view channel-chunk size for tokenization; must divide the
      view's segment width.
    '''

    def __init__(self, view_splits=((800,), (800,), (480, 480)),
                 chunks=(8, 8, 8),
                 d_model=96, n_heads=4, n_layers=2, band=1, ffn_mult=4,
                 n_input_channels=1, n_classes=1,
                 unet_checkpoints=None, freeze_unets=False,
                 init_checkpoint=None,
                 use_checkpoint=True, checkpoint_trunks=False,
                 attn_mode='all'):
        super().__init__()

        # Activation checkpointing (recompute in backward) to fit full-plane
        # training images in GPU memory.  use_checkpoint covers the attention
        # blocks and fusion convs, which are batch-norm free and so safe to
        # run twice.  checkpoint_trunks extends it to the UNet trunks: saves
        # several GB more but double-updates their batch-norm running stats.
        self.use_checkpoint = bool(use_checkpoint)
        self.checkpoint_trunks = bool(checkpoint_trunks)

        view_splits = [ [int(w) for w in vs] for vs in view_splits ]
        chunks = [int(c) for c in chunks]
        if len(view_splits) != len(chunks):
            raise ValueError(f'got {len(view_splits)} view_splits but {len(chunks)} chunks')
        for vs, chunk in zip(view_splits, chunks):
            if len(set(vs)) != 1:
                raise ValueError(f'segments of one view must have equal width, got {vs}')
            if vs[0] % chunk:
                raise ValueError(f'chunk {chunk} does not divide segment width {vs[0]}')

        self.view_splits = view_splits
        self.view_totals = [sum(vs) for vs in view_splits]
        self.chunks = chunks
        self.nviews = len(view_splits)

        self.trunks = nn.ModuleList()
        self.embeds = nn.ModuleList()
        self.expands = nn.ModuleList()
        self.fuses = nn.ModuleList()
        self.heads = nn.ModuleList()
        self.gammas = nn.ParameterList()


        '''
        Create the UNets + the layers to embed UNet output into tokens and also
        the learned gates (gammas) which "turn on" the effects from the attention
        mechanism
        '''
        self.ntok_per_seg = list()   # flat, in (view, segment) order
        nseg_total = 0
        for vs, chunk in zip(view_splits, chunks):
            trunk = UNet(n_channels=n_input_channels, n_classes=n_classes,
                         batch_norm=True, bilinear=True, padding=True)
            feat = trunk.out_features
            self.trunks.append(trunk)
            self.embeds.append(nn.Conv2d(feat, d_model, kernel_size=(chunk, 1),
                                         stride=(chunk, 1)))
            self.expands.append(nn.ConvTranspose2d(d_model, feat,
                                                   kernel_size=(chunk, 1),
                                                   stride=(chunk, 1)))
            # no batch norm in the correction path: it must behave identically
            # at any batch size and stay inert while gamma is zero.
            self.fuses.append(dconv(2*feat, feat, padding=1, batch_norm=False))
            self.heads.append(nn.Conv2d(feat, n_classes, 1))
            self.gammas.append(nn.Parameter(torch.zeros(1)))
            for w in vs:
                self.ntok_per_seg.append(w // chunk)
            nseg_total += len(vs)

        ntok = sum(self.ntok_per_seg)

        '''
        Crate learned embeddings from the token positions (a function of view + channel)
        and also the 'segment' which is a redundant function of view and channel
        '''
        self.pos_embed = nn.Parameter(torch.empty(1, d_model, ntok, 1).normal_(std=0.02))
        self.seg_embeds = nn.ParameterList(
            nn.Parameter(torch.empty(1, d_model, 1, 1).normal_(std=0.02))
            for _ in range(nseg_total))

        '''
        Attention blocks
        '''
        self.blocks = nn.ModuleList(
            BandedAttentionBlock(d_model, n_heads, band=band, ffn_mult=ffn_mult)
            for _ in range(n_layers))

        # Token spans per segment, tagged with the owning view, in the same
        # flat (view, segment) order tokens are concatenated.  The blocks need
        # this to restrict attention by segment or view.
        self.token_segments = list()
        at = iseg = 0
        for iv, vs in enumerate(view_splits):
            for _ in vs:
                n = self.ntok_per_seg[iseg]
                self.token_segments.append((at, at + n, iv))
                at += n
                iseg += 1
        for blk in self.blocks:
            blk.segments = self.token_segments
        # Default to 'all': two faces of one view are separate drift regions
        # and must not attend to each other.  'legacy' allows that and exists
        # only to reproduce models trained before modes were added.
        self.set_attention_mode(attn_mode)

        '''
        Loading different portions -- could do with some protection against double loading?
        '''
        if unet_checkpoints:
            if len(unet_checkpoints) != self.nviews:
                raise ValueError(f'need {self.nviews} unet_checkpoints, '
                                 f'got {len(unet_checkpoints)}')
            for trunk, path in zip(self.trunks, unet_checkpoints):
                self.load_trunk_checkpoint(trunk, path)

        self._freeze_unets = bool(freeze_unets)
        if self._freeze_unets:
            for trunk in self.trunks:
                trunk.requires_grad_(False)
                trunk.eval()  # also stop batch-norm stat updates, see train()

        if init_checkpoint:
            self.load_full_checkpoint(init_checkpoint)

    @staticmethod
    def load_trunk_checkpoint(trunk, path):
        '''
        Warm-start one trunk from a dnnroi_custom-style checkpoint whose model
        keys are prefixed "unet." (or from a bare UNet state dict).
        '''
        cp = torch.load(path, map_location='cpu', weights_only=True)
        sd = cp.get('model_state_dict', cp)
        stripped = { (k[len('unet.'):] if k.startswith('unet.') else k): v
                     for k, v in sd.items() }
        trunk.load_state_dict(stripped, strict=True)
        log.info(f'xvunet: loaded trunk weights from {path}')

    def load_full_checkpoint(self, path):
        '''
        Load a full XViewUNet model_state_dict, ignoring any optimizer state.
        Used to continue from a frozen-trunk stage-1 checkpoint when unfreezing.
        Accepts checkpoints of the app Network wrapper (keys prefixed "xvunet.").
        '''
        cp = torch.load(path, map_location='cpu', weights_only=True)
        sd = cp.get('model_state_dict', cp)
        sd = { (k[len('xvunet.'):] if k.startswith('xvunet.') else k): v
               for k, v in sd.items() }
        self.load_state_dict(sd, strict=True)
        log.info(f'xvunet: loaded full model weights from {path}')

    def set_attention_mode(self, mode):
        '''
        Set the attention scope on every block; see ATTN_MODES.

        Nothing here changes parameter shapes, so a single trained checkpoint
        can be evaluated under each mode in turn.  Note the default is
        'legacy', which alone lets the two faces of a view attend to each
        other -- 'all' forbids that, so the two differ on a multi-segment view
        and a model trained under one is not the same function under the other.
        '''
        if mode not in ATTN_MODES:
            raise ValueError(f'unknown attn_mode {mode!r}, want one of '
                             f'{list(ATTN_MODES)}')
        if mode != 'legacy' and not self.token_segments:
            raise RuntimeError('no segment layout to restrict attention with')
        for blk in self.blocks:
            blk.attn_mode = mode
        log.info(f'xvunet: attention mode {mode}')
        return self

    def train(self, mode=True):
        '''
        As nn.Module.train() but frozen trunks stay in eval mode so their
        batch-norm running statistics are not perturbed.
        '''
        super().train(mode)
        if self._freeze_unets:
            for trunk in self.trunks:
                trunk.eval()
        return self

    def _ckpt(self, enabled, fn, *args):
        '''
        Run fn(*args), under activation checkpointing when enabled.  Skipped
        unless training with grad, where alone the recompute pays for itself.
        '''
        if enabled and self.training and torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, x):
        '''
        (B, n_input_channels, sum(view_totals), T) in, n_classes logits out at
        the same channel and tick resolution.

        Three passes over the (view, segment) pairs: trunks produce full-res
        features and their chunked tokens, the tokens attend across views, then
        each segment's attended tokens are expanded back to full resolution and
        fused with its features.  The final logits are the plain per-view UNet
        output plus a gamma-gated correction, so gamma=0 reproduces the trunks.
        '''
        views = torch.split(x, self.view_totals, dim=2)

        '''Run through 'trunks' i.e. UNets and create tokens'''
        feats = list()      # per (view, segment) full-res trunk features
        tokens = list()     # per (view, segment) token maps (B, d, ntok, T)
        iseg = 0
        for iv, view in enumerate(views):
            for seg in torch.split(view, self.view_splits[iv], dim=2):
                f = self._ckpt(self.checkpoint_trunks,
                               self.trunks[iv].forward_features, seg)
                feats.append(f)                              # (B, F, Cseg, T)
                tok = self.embeds[iv](f) + self.seg_embeds[iseg]
                tokens.append(tok)
                iseg += 1

        '''Run through attention'''
        tok = torch.cat(tokens, dim=2) + self.pos_embed      # (B, d, N, T)
        tok = tok.permute(0, 3, 2, 1)                        # (B, T, N, d)
        for blk in self.blocks:
            tok = self._ckpt(self.use_checkpoint, blk, tok)
        tok = tok.permute(0, 3, 2, 1)                        # (B, d, N, T)

        outs = list()
        iseg = 0
        itok = 0
        for iv in range(self.nviews):
            for _ in self.view_splits[iv]:
                ntok = self.ntok_per_seg[iseg]
                seg_tok = tok.narrow(2, itok, ntok)
                up = self.expands[iv](seg_tok)               # (B, F, Cseg, T)
                fused = self._ckpt(self.use_checkpoint, self.fuses[iv],
                                   torch.cat((feats[iseg], up), dim=1))
                logits = self.trunks[iv].segmap(feats[iseg]) \
                    + self.gammas[iv] * self.heads[iv](fused)
                outs.append(logits)
                iseg += 1
                itok += ntok

        return torch.cat(outs, dim=2)
