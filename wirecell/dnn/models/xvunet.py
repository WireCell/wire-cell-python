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
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .unet import UNet, dconv

import logging
log = logging.getLogger("wirecell.dnn")


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

    def _band_mask(self, T, N, device):
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
        # x: (B, T, N, d)
        B, T, N, d = x.shape
        h = self.n_heads
        dh = d // h

        y = self.norm1(x)
        q, k, v = self.qkv(y).chunk(3, dim=-1)

        # keys/values from ticks t-band .. t+band: roll(k, -s)[t] == k[t+s]
        shifts = range(-self.band, self.band + 1)
        kb = torch.cat([torch.roll(k, -s, dims=1) for s in shifts], dim=2)
        vb = torch.cat([torch.roll(v, -s, dims=1) for s in shifts], dim=2)
        M = kb.shape[2]

        q = q.reshape(B*T, N, h, dh).transpose(1, 2)     # (B*T, h, N, dh)
        kb = kb.reshape(B*T, M, h, dh).transpose(1, 2)   # (B*T, h, M, dh)
        vb = vb.reshape(B*T, M, h, dh).transpose(1, 2)

        mask = self._band_mask(T, N, x.device)           # (T, M)
        mask = mask.repeat(B, 1).view(B*T, 1, 1, M)

        o = F.scaled_dot_product_attention(q, kb, vb, attn_mask=mask)
        o = o.transpose(1, 2).reshape(B, T, N, d)

        x = x + self.gamma1 * self.proj(o)
        x = x + self.gamma2 * self.ffn(self.norm2(x))
        return x


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
                 use_checkpoint=True, checkpoint_trunks=False):
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
        self.pos_embed = nn.Parameter(torch.empty(1, d_model, ntok, 1).normal_(std=0.02))
        self.seg_embeds = nn.ParameterList(
            nn.Parameter(torch.empty(1, d_model, 1, 1).normal_(std=0.02))
            for _ in range(nseg_total))

        self.blocks = nn.ModuleList(
            BandedAttentionBlock(d_model, n_heads, band=band, ffn_mult=ffn_mult)
            for _ in range(n_layers))

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
        if enabled and self.training and torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, x):
        # x: (B, n_input_channels, sum(view_totals), T)
        views = torch.split(x, self.view_totals, dim=2)

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
