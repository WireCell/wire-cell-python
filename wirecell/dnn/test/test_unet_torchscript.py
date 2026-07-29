#!/usr/bin/env python
'''
Guards for the UNet properties that the TorchScript export depends on.

These invariants are easy to break by accident and fail quietly, so they are
pinned here:

- the model can be scripted at all (a plain Python list of submodules, an
  untyped list, or an attribute that changes type will all break this),
- forward_features stays reachable on the scripted module (xvunet calls it),
- state dicts written before the legs became nn.ModuleLists still load,
- the *positional* parameter order does not move, because an optimizer state
  dict references parameters by position -- reordering silently drops momentum
  buffers onto the wrong tensors on resume,
- the pad/crop arithmetic is recomputed per call, so one input size does not
  poison a later different one.
'''

import pytest

torch = pytest.importorskip("torch")

from wirecell.dnn.models.unet import UNet


# The geometries the apps actually build, plus the two edge cases no app covers.
# (bilinear=False with padding=True is a pre-existing channel-mismatch bug in
# make_umerge and is deliberately not listed.)
GEOMETRIES = [
    pytest.param(dict(n_channels=3, n_classes=1, batch_norm=True, bilinear=True,
                      padding=True), (1, 3, 100, 150), id='dnnroi'),
    pytest.param(dict(n_channels=1, n_classes=1, batch_norm=True, bilinear=True,
                      padding=True), (1, 1, 100, 150), id='dnnroi_custom_c1'),
    pytest.param(dict(n_channels=1, n_classes=2, batch_norm=True, bilinear=True,
                      padding=True), (1, 1, 100, 150), id='dnnroi_regres'),
    pytest.param(dict(n_channels=3, n_classes=6, batch_norm=False, bilinear=False,
                      padding=False), (1, 3, 572, 572), id='crop_paper_default'),
    pytest.param(dict(n_channels=3, n_classes=1, nskips=3, batch_norm=True,
                      bilinear=True, padding=True), (1, 3, 100, 150), id='nskips3'),
]


def _legacy_keys(state_dict, nskips):
    '''
    Rewrite a current state dict into the pre-ModuleList layout, i.e. the inverse
    of UNet._remap_legacy_keys.  Lets us test the migration without shipping a
    binary fixture.
    '''
    out = dict()
    for key, val in state_dict.items():
        parts = key.split('.')
        if parts[0] == 'down_blocks':
            idx, kind = int(parts[1]), parts[2]
            out['.'.join([f'down_{"dconv" if kind == "dconv" else "dsamp"}_{idx}']
                         + parts[3:])] = val
        elif parts[0] == 'up_blocks':
            idx, kind = int(parts[1]), parts[2]
            out['.'.join([f'up_{kind}_{nskips - 1 - idx}'] + parts[3:])] = val
        else:
            out[key] = val
    return out


@pytest.mark.parametrize("kwargs,shape", GEOMETRIES)
def test_scriptable_and_matches_eager(kwargs, shape):
    net = UNet(**kwargs).eval()
    x = torch.randn(*shape)
    with torch.no_grad():
        want = net(x)
    scripted = torch.jit.script(net)
    with torch.no_grad():
        got = scripted(x)
    assert torch.equal(got, want)


@pytest.mark.parametrize("kwargs,shape", GEOMETRIES)
def test_forward_features_exported(kwargs, shape):
    'xvunet calls forward_features directly, so it must survive scripting.'
    net = UNet(**kwargs).eval()
    x = torch.randn(*shape)
    scripted = torch.jit.script(net)
    assert hasattr(scripted, 'forward_features')
    with torch.no_grad():
        assert torch.equal(scripted.forward_features(x), net.forward_features(x))
    assert net.out_features > 0


@pytest.mark.parametrize("kwargs,shape", GEOMETRIES)
def test_legacy_state_dict_loads(kwargs, shape):
    nskips = kwargs.get('nskips', 4)
    ref = UNet(**kwargs).eval()
    x = torch.randn(*shape)
    with torch.no_grad():
        want = ref(x)

    legacy = _legacy_keys(ref.state_dict(), nskips)
    assert any(k.startswith(('down_dconv_', 'up_dconv_')) for k in legacy)

    net = UNet(**kwargs).eval()
    missing, unexpected = net.load_state_dict(legacy, strict=False)
    assert not missing and not unexpected
    with torch.no_grad():
        assert torch.equal(net(x), want)


@pytest.mark.parametrize("kwargs,shape", GEOMETRIES)
def test_positional_parameter_order_is_stable(kwargs, shape):
    '''
    An optimizer_state_dict maps buffers to parameters by position.  Pin the
    order so a future restructuring cannot silently corrupt a resumed run.
    '''
    net = UNet(**kwargs)
    names = [n for n, _ in net.named_parameters()]
    # down leg ascending, then bottom, then up leg, then segmap
    stages = []
    for n in names:
        top = n.split('.')[0]
        if not stages or stages[-1] != top:
            stages.append(top)
    expected = ['down_blocks', 'bottom', 'up_blocks', 'segmap']
    assert stages == expected, f'parameter groups out of order: {stages}'
    assert names[-1] == 'segmap.bias'


def test_pads_not_cached_across_sizes():
    '''
    The pad amounts used to be computed once and reused, so a model that had seen
    one geometry produced wrong results for another.  Interleave two sizes and
    require the repeated one to be unchanged.
    '''
    kwargs = dict(n_channels=1, n_classes=1, batch_norm=True, bilinear=True,
                  padding=True)
    net = UNet(**kwargs).eval()
    a_in = torch.randn(1, 1, 100, 150)
    b_in = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        a1 = net(a_in)
        b = net(b_in)
        a2 = net(a_in)
    assert torch.equal(a1, a2)
    assert b.shape == (1, 1, 64, 64)
    # and a freshly built model that has only ever seen the second size agrees
    fresh = UNet(**kwargs).eval()
    fresh.load_state_dict(net.state_dict())
    with torch.no_grad():
        assert torch.equal(fresh(b_in), b)
