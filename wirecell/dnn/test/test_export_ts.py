#!/usr/bin/env python
'''
Tests for the TorchScript export helpers behind `wcpy dnn export-ts`.

The reload checks matter most: a .ts file is only useful if torch.jit.load can
run it without the defining Python classes, and if it reproduces the eager model.
'''

import pytest

torch = pytest.importorskip("torch")

from wirecell.dnn import io as dio
from wirecell.dnn.models.unet import UNet

SHAPE = (1, 1, 64, 96)


def _net():
    torch.manual_seed(1234)
    return UNet(n_channels=1, n_classes=1, batch_norm=True, bilinear=True,
                padding=True).eval()


def _x(shape=SHAPE):
    torch.manual_seed(7)
    return torch.randn(*shape)


@pytest.mark.parametrize("method", ['script', 'trace'])
def test_export_reload_matches_eager(tmp_path, method):
    net = _net()
    x = _x()
    with torch.no_grad():
        want = net(x)

    out = tmp_path / f'model_{method}.ts'
    info = dio.save_torchscript(net, out, shape=SHAPE, method=method)

    assert out.exists()
    assert info['method'] == method
    assert info['max_abs_diff'] == 0.0
    assert info['out_shape'] == tuple(want.shape)

    # reload the saved file, as a consumer would
    loaded = torch.jit.load(str(out))
    with torch.no_grad():
        got = loaded(x)
    assert torch.equal(got, want)


def test_scripted_export_is_shape_generic(tmp_path):
    'Scripting keeps the pad arithmetic dynamic, so other sizes still work.'
    net = _net()
    out = tmp_path / 'generic.ts'
    dio.save_torchscript(net, out, shape=SHAPE, method='script')
    loaded = torch.jit.load(str(out))
    for shape in [(2, 1, 64, 96), (1, 1, 100, 150), (1, 1, 32, 32)]:
        with torch.no_grad():
            got = loaded(torch.zeros(*shape))
        assert got.shape[0] == shape[0]
        assert got.shape[2:] == shape[2:]


def test_sigmoid_wrapping(tmp_path):
    net = _net()
    x = _x()
    with torch.no_grad():
        want = torch.sigmoid(net(x))

    out = tmp_path / 'sig.ts'
    info = dio.save_torchscript(net, out, shape=SHAPE, method='script', sigmoid=True)
    assert info['sigmoid'] is True
    assert 0.0 <= info['out_min'] and info['out_max'] <= 1.0

    loaded = torch.jit.load(str(out))
    with torch.no_grad():
        assert torch.equal(loaded(x), want)


def test_trace_requires_a_shape(tmp_path):
    with pytest.raises(ValueError):
        dio.save_torchscript(_net(), tmp_path / 'x.ts', shape=None, method='trace')


def test_unknown_method_rejected(tmp_path):
    with pytest.raises(ValueError):
        dio.save_torchscript(_net(), tmp_path / 'x.ts', shape=SHAPE, method='onnx')


def test_export_without_shape_is_unverified(tmp_path):
    'No shape means no example input, so there is nothing to compare against.'
    out = tmp_path / 'unverified.ts'
    info = dio.save_torchscript(_net(), out, shape=None, method='script')
    assert out.exists()
    assert 'max_abs_diff' not in info


def test_export_leaves_model_in_eval_mode(tmp_path):
    '''
    BatchNorm must use running statistics in the exported graph, so the model is
    switched to eval() before conversion.
    '''
    net = _net()
    net.train()
    dio.save_torchscript(net, tmp_path / 'x.ts', shape=SHAPE, method='script')
    assert not net.training


@pytest.mark.parametrize("flavour", ['checkpoint', 'bare', 'ddp'])
def test_load_model_state_flavours(tmp_path, flavour):
    'A checkpoint, a bare state dict, and a DDP-prefixed one all load.'
    src = _net()
    sd = src.state_dict()
    path = tmp_path / f'{flavour}.pt'
    if flavour == 'checkpoint':
        torch.save(dict(model_state_dict=sd, optimizer_state_dict={}, runs={}), path)
    elif flavour == 'bare':
        torch.save(sd, path)
    else:
        torch.save(dict(model_state_dict={f'module.{k}': v for k, v in sd.items()}),
                   path)

    dst = UNet(n_channels=1, n_classes=1, batch_norm=True, bilinear=True,
               padding=True).eval()
    dio.load_model_state(path, dst)

    x = _x()
    with torch.no_grad():
        assert torch.equal(dst(x), src(x))


def test_load_model_state_returns_extra_entries(tmp_path):
    path = tmp_path / 'ckpt.pt'
    torch.save(dict(model_state_dict=_net().state_dict(),
                    optimizer_state_dict={}, runs={0: 'x'}, epochs={}), path)
    dst = _net()
    rest = dio.load_model_state(path, dst)
    assert 'runs' in rest and 'epochs' in rest
    # the state dicts themselves are consumed, not handed back
    assert 'model_state_dict' not in rest
    assert 'optimizer_state_dict' not in rest


#
# xvunet's banded attention mask must ride along with the module, not be baked
# into the traced graph.  See wcpy-oci.
#

XV_CFG = dict(view_splits='[[80],[80],[48,48]]', chunks='[8,8,8]',
              d_model='32', n_heads='4', n_layers='2', band='1')
XV_SHAPE = (1, 1, 80 + 80 + 48 + 48, 64)


def _xvunet():
    from wirecell.dnn.apps.xvunet.model import Network
    torch.manual_seed(0)
    return Network(**XV_CFG).eval()


def test_band_mask_stays_out_of_state_dict():
    '''
    The masks are registered so they move with the module, but non-persistently
    so they never reach state_dict -- otherwise every checkpoint written before
    this change would fail to load with strict=True, and vice versa.
    '''
    net = _xvunet()
    before = set(net.state_dict().keys())

    with torch.no_grad():
        net(_x(XV_SHAPE))

    masks = [n for n, _ in net.named_buffers() if '_band_mask_' in n]
    assert masks, 'the forward should have registered at least one mask buffer'
    assert set(net.state_dict().keys()) == before


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='needs two GPUs')
def test_traced_xvunet_runs_on_another_device(tmp_path):
    '''
    A .ts traced on one GPU must load and run on a different one.  Tracing bakes
    any tensor that is not a parameter or buffer into the graph at the device it
    was built on, so a mask held in a plain dict left the file stuck on cuda:0,
    failing elsewhere with "two devices ... argument attn_bias".
    '''
    net = _xvunet().to('cuda:0')
    x = _x(XV_SHAPE)
    path = tmp_path / 'xv.ts'
    dio.save_torchscript(net, path, shape=XV_SHAPE, method='trace',
                         device='cuda:0')

    outs = dict()
    for dev in ('cuda:0', 'cuda:1'):
        mod = torch.jit.load(str(path), map_location=dev)
        with torch.no_grad():
            outs[dev] = mod(x.to(dev)).cpu()
    # same graph, same constants, only the device differs
    assert torch.equal(outs['cuda:0'], outs['cuda:1'])
