#!/usr/bin/env python
'''
File I/O


https://pytorch.org/tutorials/beginner/saving_loading_models.html
'''

import torch

import logging
log = logging.getLogger("wirecell.dnn")


def save_checkpoint(path, model, optimizer, **kwds):
    '''
    Save a checkpoint to file at path.

    Checkpoint consists of model and optimizer state dicts and any additional
    attributes supplied as kwds.
    '''
    kwds.update(model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict())
    torch.save(kwds, path)


def load_checkpoint_raw(path):
    return torch.load(path, weights_only=True)
    

def load_checkpoint_from(cp, model, optimizer):
    '''
    As load_checkpoint() but from an already-loaded checkpoint dict.

    Resuming needs the checkpoint twice: once before the model is built, to
    reconcile it against the config, and again to restore state.  These files
    run to hundreds of MB, so read once and pass the dict here.  The caller's
    dict is left intact.
    '''
    cp = dict(cp)
    model.load_state_dict(cp.pop("model_state_dict"))
    optimizer.load_state_dict(cp.pop("optimizer_state_dict"))
    return cp


def load_checkpoint(path, model, optimizer):
    '''
    Load a checkpoint.

    The model and optimizer state dicts are updated and a dict of any additional
    parameters is returned.
    '''
    return load_checkpoint_from(load_checkpoint_raw(path), model, optimizer)


def checkpoint_model_args(cp, structural_keys=()):
    '''
    Return the model configuration a checkpoint records for its most recent run,
    or None if it records none.

    Newer checkpoints nest it under "model_args".  Older ones spread it flat
    into the run dict, as siblings of ntrain/batch/name, so as a fallback the
    named structural_keys are picked back out of the run dict -- enough to
    validate a resume, which is all a caller wants it for.  None of the keys a
    model declares structural collide with the run metadata.

    Returning None means "unknown", not "empty": a caller must not read it as
    the checkpoint having been trained with default settings.
    '''
    if not isinstance(cp, dict):
        return None
    runs = cp.get("runs") or dict()
    if not runs:
        return None
    run = runs[max(runs.keys())]
    if not isinstance(run, dict):
        return None

    args = run.get("model_args")
    if args is not None:
        return dict(args)

    args = {k: run[k] for k in structural_keys if k in run}
    if args:
        log.debug(f'checkpoint has no "model_args"; recovered '
                  f'{sorted(args)} from the flat (pre-nesting) run record')
        return args
    return None


def load_model_state(path, model, strict=True):
    '''
    Load only the model weights from path into model.  Returns whatever other
    entries the file held (empty if it was a bare state dict).

    Unlike load_checkpoint() this needs no optimizer, which is what inference and
    export want.  The file may be either a checkpoint as written by
    save_checkpoint() or a bare state dict from torch.save(model.state_dict()).
    A "module." prefix on every key, as left by DistributedDataParallel, is
    stripped.
    '''
    return load_model_state_from(load_checkpoint_raw(path), model, strict=strict,
                                 path=path)


def load_model_state_from(obj, model, strict=True, path='<loaded>'):
    '''
    As load_model_state() but from an already-loaded object, so a caller that
    had to read the file early (to reconcile it against a config) need not read
    it again.  path is used only in log messages.
    '''
    rest = dict()
    if isinstance(obj, dict) and "model_state_dict" in obj:
        rest = dict(obj)
        sd = rest.pop("model_state_dict")
        rest.pop("optimizer_state_dict", None)
        log.debug(f'{path}: checkpoint, extra keys {sorted(rest.keys())}')
    else:
        sd = obj
        log.debug(f'{path}: bare state dict')

    nmod = sum(1 for k in sd if k.startswith("module."))
    if nmod:
        log.info(f'{path}: stripping "module." prefix from {nmod} keys '
                 '(DistributedDataParallel checkpoint)')
        sd = {(k[len("module."):] if k.startswith("module.") else k): v
              for k, v in sd.items()}

    model.load_state_dict(sd, strict=strict)
    return rest


class _Sigmoid(torch.nn.Module):
    '''
    Wrap a network so its output passes through a sigmoid.

    Some apps (dnnroi_custom) return raw logits because they train against
    BCEWithLogitsLoss, while a consumer of the exported model usually wants a
    probability.  Applying it here keeps the training graph untouched.
    '''

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


def save_torchscript(model, path, shape=None, method='script', device='cpu',
                     sigmoid=False, seed=0):
    '''
    Convert model to TorchScript and write it to path.  Returns an info dict.

    - method :: 'script' compiles the model without needing an example input and
      keeps it shape-generic.  'trace' records one execution and is the fallback
      for a model that will not compile.
    - shape :: input shape used to trace with, and (for either method) to verify
      the converted module against the original.  Required for 'trace'.
    - sigmoid :: wrap the output in a sigmoid before converting.
    - device :: where to build and convert.  Note a model converted on CUDA bakes
      the device into its graph and will then require a GPU to load.

    The model is switched to eval() first: these networks use BatchNorm, so
    exporting in train mode would bake in batch-statistics behaviour rather than
    the learned running statistics.
    '''
    if sigmoid:
        model = _Sigmoid(model)

    model = model.to(device)
    model.eval()

    example = None
    reference = None
    if shape is not None:
        gen = torch.Generator(device='cpu').manual_seed(seed)
        example = torch.randn(*shape, generator=gen).to(device)
        with torch.no_grad():
            reference = model(example)

    if method == 'trace':
        if example is None:
            raise ValueError("method='trace' needs an input shape")
        with torch.no_grad():
            tsmod = torch.jit.trace(model, example)
    elif method == 'script':
        tsmod = torch.jit.script(model)
    else:
        raise ValueError(f'unknown method {method!r}, want "script" or "trace"')

    info = dict(method=method, device=str(device), sigmoid=bool(sigmoid),
                path=str(path), shape=None if shape is None else tuple(shape))

    if example is not None:
        with torch.no_grad():
            got = tsmod(example)
        info['out_shape'] = tuple(got.shape)
        info['max_abs_diff'] = float((got - reference).abs().max())
        info['out_min'] = float(got.min())
        info['out_max'] = float(got.max())

    tsmod.save(str(path))
    return info

