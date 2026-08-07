#!/usr/bin/env python
'''
Tests for the xvunet model and app wrapper.

All tests run on CPU with small tensors.
'''

import pytest
import torch

from wirecell.dnn.models.unet import UNet
from wirecell.dnn.models.xvunet import (XViewUNet, BandedAttentionBlock,
                                        ATTN_MODES)
from wirecell.dnn.apps.xvunet.model import Network


VIEW_SPLITS = [[80], [80], [48, 48]]
CHUNKS = [8, 8, 8]
TOTAL = 80 + 80 + 48 + 48
T = 64


def small_model(**kwds):
    args = dict(view_splits=VIEW_SPLITS, chunks=CHUNKS,
                d_model=32, n_heads=4, n_layers=2, band=1)
    args.update(kwds)
    return XViewUNet(**args)


def test_forward_shape():
    model = small_model().eval()
    x = torch.rand(2, 1, TOTAL, T)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, TOTAL, T)


def test_zero_init_equals_trunks():
    '''
    At construction the attention path is gated off: the model output must be
    exactly the per-view/per-segment UNet trunk outputs.
    '''
    model = small_model().eval()
    x = torch.rand(1, 1, TOTAL, T)
    with torch.no_grad():
        y = model(x)

        parts = list()
        views = torch.split(x, model.view_totals, dim=2)
        for iv, view in enumerate(views):
            for seg in torch.split(view, model.view_splits[iv], dim=2):
                parts.append(model.trunks[iv](seg))
        ref = torch.cat(parts, dim=2)

    assert torch.allclose(y, ref, atol=0, rtol=0)


def test_chunk_divisibility_error():
    with pytest.raises(ValueError):
        small_model(chunks=[7, 8, 8])


def test_unequal_segments_error():
    with pytest.raises(ValueError):
        small_model(view_splits=[[80], [80], [48, 40]])


def test_banded_locality():
    '''
    A perturbation at tick t0 may only influence output ticks within
    +/- band per attention layer, and edge ticks must not wrap around.

    The perturbation must be non-uniform across the feature axis: norm1 is a
    LayerNorm, so adding a constant to every feature of a token shifts only its
    mean and is removed exactly.  With a uniform perturbation the neighbouring
    ticks move by ~1e-7 of float rounding rather than by attention, and this
    test then passes even with the attention gate shut.  The tolerance and the
    lower-bound assertion below exist for the same reason.
    '''
    torch.manual_seed(0)
    band = 1
    block = BandedAttentionBlock(d_model=16, n_heads=2, band=band).eval()
    # open the attention gate so the test is not vacuous
    block.gamma1.data.fill_(1.0)

    Tt, N = 12, 6
    x = torch.rand(1, Tt, N, 16)
    pert = torch.randn(N, 16) * 3.0

    for t0 in (0, 6, Tt - 1):
        x2 = x.clone()
        x2[:, t0] += pert
        with torch.no_grad():
            d = (block(x2) - block(x)).abs().amax(dim=(0, 2, 3))  # per tick
        affected = set(torch.nonzero(d > 1e-4).flatten().tolist())
        allowed = set(range(max(0, t0 - band), min(Tt, t0 + band + 1)))
        assert affected <= allowed
        # every in-band neighbour must actually be reached, or attention being
        # silently disabled would still satisfy the containment above
        assert affected == allowed


def _open_gates(model, gate=0.5):
    '''Open every gate so attention-path differences reach the output.'''
    with torch.no_grad():
        for g in model.gammas:
            g.fill_(0.7)
        for blk in model.blocks:
            blk.gamma1.fill_(gate)
            blk.gamma2.fill_(gate)
    return model


def test_attn_mode_reachability():
    '''
    Each mode admits exactly the token pairs its rule allows.  Perturbing one
    token of the first W face, the queries whose output moves must be:

      legacy  every other token
      all     own segment plus every segment of another view (never the
              sibling W face)
      intra   own segment only
      inter   segments of another view only
      none    nothing
    '''
    torch.manual_seed(0)
    d, Tt = 8, 6
    segs = [(0, 4, 0), (4, 8, 1), (8, 12, 2), (12, 16, 2)]   # U, V, W0, W1
    N = 16
    block = BandedAttentionBlock(d_model=d, n_heads=2, band=1).eval()
    block.segments = segs
    with torch.no_grad():
        block.gamma1.fill_(1.0)
        block.gamma2.zero_()          # isolate the attention branch

    x = torch.rand(1, Tt, N, d)
    pert = torch.randn(Tt, d) * 3.0   # non-uniform, see test_banded_locality
    p = 8                             # first token of W face 0

    def reached(mode):
        block.attn_mode = mode
        with torch.no_grad():
            y0 = block(x)
            x2 = x.clone()
            x2[0, :, p, :] += pert
            y1 = block(x2)
        diff = (y1 - y0).abs().amax(dim=3)[0].amax(dim=0)
        return {i for i in range(N) if diff[i] > 1e-5 and i != p}

    assert reached('legacy') == set(range(N)) - {p}
    assert reached('all') == set(range(12)) - {p}
    assert reached('intra') == set(range(8, 12)) - {p}
    assert reached('inter') == set(range(8))
    assert reached('none') == set()


def test_all_equals_legacy_without_multisegment_view():
    '''
    'all' differs from 'legacy' only by same-view-different-segment pairs, so
    with one segment per view the two must agree bit for bit.
    '''
    torch.manual_seed(0)
    model = _open_gates(
        XViewUNet(view_splits=[[32], [32], [32]], chunks=[8, 8, 8],
                  d_model=16, n_heads=2).eval())
    x = torch.rand(1, 1, 96, 32)
    with torch.no_grad():
        legacy = model.set_attention_mode('legacy')(x)
        every = model.set_attention_mode('all')(x)
    assert torch.equal(legacy, every)


def test_all_differs_from_legacy_on_two_faced_view():
    '''The two W faces may see each other under 'legacy' but not under 'all'.'''
    torch.manual_seed(0)
    model = _open_gates(small_model().eval())
    x = torch.rand(1, 1, TOTAL, T)
    with torch.no_grad():
        legacy = model.set_attention_mode('legacy')(x)
        every = model.set_attention_mode('all')(x)
    assert not torch.allclose(legacy, every)


def test_attn_mode_leaves_state_dict_alone():
    '''
    Switching mode must not touch parameters, so one trained checkpoint can be
    evaluated under every mode.
    '''
    model = small_model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    for mode in ATTN_MODES:
        model.set_attention_mode(mode)
    after = model.state_dict()
    assert set(before) == set(after)
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_attn_mode_unknown():
    with pytest.raises(ValueError):
        small_model().set_attention_mode('sideways')


def test_warm_start_trunks(tmp_path):
    '''
    A dnnroi_custom-style checkpoint ("unet."-prefixed UNet keys) loads into
    the trunks.
    '''
    ref = UNet(n_channels=1, n_classes=1,
               batch_norm=True, bilinear=True, padding=True)
    path = tmp_path / 'pre.pt'
    torch.save({'model_state_dict':
                {'unet.' + k: v for k, v in ref.state_dict().items()}}, path)

    model = small_model(unet_checkpoints=[str(path)] * 3)
    for trunk in model.trunks:
        for k, v in trunk.state_dict().items():
            assert torch.equal(v, ref.state_dict()[k]), k


def test_freeze_unets(tmp_path):
    model = small_model(freeze_unets=True)
    for trunk in model.trunks:
        assert not any(p.requires_grad for p in trunk.parameters())
    assert any(p.requires_grad for p in model.parameters())

    # frozen trunks stay in eval mode under net.train() (batch-norm stats)
    model.train()
    assert model.training
    for trunk in model.trunks:
        assert not trunk.training
    for blk in model.blocks:
        assert blk.training


def test_train_step_with_checkpointing():
    '''
    A full loss/backward/step runs with activation checkpointing enabled and
    the attention-path gates receive gradient (so the path can start learning
    even though it is zero-gated at init).
    '''
    torch.manual_seed(0)
    model = small_model(use_checkpoint=True, checkpoint_trunks=True)
    model.train()
    x = torch.rand(2, 1, TOTAL, T)
    y = (torch.rand(2, 1, TOTAL, T) > 0.5).float()

    crit = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = crit(model(x), y)
    assert torch.isfinite(loss)
    loss.backward()

    for g in model.gammas:
        assert g.grad is not None and g.grad.abs().sum() > 0
    # attention params participate in the graph (grads exist; zero until the
    # head gates move off zero)
    assert model.blocks[0].qkv.weight.grad is not None
    assert model.trunks[0].segmap.weight.grad.abs().sum() > 0
    opt.step()


def test_network_wrapper_and_init_checkpoint(tmp_path):
    '''
    The app Network parses INI-string config and init_checkpoint restores a
    full (Network-prefixed) checkpoint.
    '''
    cfg = dict(view_splits='[[80],[80],[48,48]]', chunks='[8,8,8]',
               d_model='32', n_heads='4', n_layers='1', band='1')
    net1 = Network(cfg)
    # make it distinguishable from a fresh init
    with torch.no_grad():
        for g in net1.xvunet.gammas:
            g.fill_(0.5)

    path = tmp_path / 'stage1.pt'
    torch.save({'model_state_dict': net1.state_dict()}, path)

    net2 = Network(dict(cfg, init_checkpoint=str(path)))
    sd1, sd2 = net1.state_dict(), net2.state_dict()
    assert sd1.keys() == sd2.keys()
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k]), k
