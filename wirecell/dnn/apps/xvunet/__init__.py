#!/usr/bin/env python
'''
The xvunet app: per-view UNet trunks + time-banded cross-view attention for
ROI finding from deconvolved images only (no MP2/MP3 input channels).

See wirecell.dnn.models.xvunet for the model.
'''
from torch import optim

## The "app" API
from .model import Network
from .data import Dataset
from wirecell.dnn.train import Classifier as Trainer
from torch.nn import BCEWithLogitsLoss as Criterion


import logging
log = logging.getLogger("wirecell.dnn")

# Per-optimizer defaults.  They differ by an order of magnitude or more, so the
# default learning rate has to follow the choice of optimizer rather than being
# a single number.
_OPT_DEFAULTS = dict(
    adamw=dict(learning_rate=1e-3, weight_decay=0.01),
    adam=dict(learning_rate=1e-3, weight_decay=0.0),
    sgd=dict(learning_rate=0.1, weight_decay=0.0005, momentum=0.9),
)


def Optimizer(params, config=None):
    '''
    Build the optimizer named by [optimizer] name, defaulting to adamw.

    AdamW is the default because this model's cross-view branch cannot be
    trained by SGD.  Its output is gated by a zero-initialised LayerScale
    (gammas), so at the start d(loss)/d(branch weights) is identically zero and
    the only gradient is gamma's own, measured at ~2e-5.  SGD's step is
    proportional to the gradient, so it stays ~1e-8 and the branch never
    activates -- the loss sits exactly at the frozen-trunk baseline.  Adam-family
    optimizers normalise per parameter, so the step size does not depend on the
    gradient scale and the gate escapes zero.  (SGD remains right for the plain
    UNet apps; this is specific to the gated residual branch.)

    Recognised keys: name, learning_rate, weight_decay, momentum (sgd only),
    beta1/beta2/eps (adam family only).
    '''
    config = config or dict()
    # skip frozen parameters (e.g. warm-started trunks with freeze_unets=True)
    params = [p for p in params if p.requires_grad]

    name = str(config.get('name', 'adamw')).strip().lower()
    if name not in _OPT_DEFAULTS:
        raise ValueError(f'unknown optimizer {name!r}, '
                         f'want one of {sorted(_OPT_DEFAULTS)}')
    dflt = _OPT_DEFAULTS[name]

    def val(key):
        return float(config.get(key, dflt[key]))

    lr, weight_decay = val('learning_rate'), val('weight_decay')

    if name == 'sgd':
        momentum = val('momentum')
        log.info(f'xvunet optimizer: SGD lr={lr} momentum={momentum} '
                 f'weight_decay={weight_decay}')
        return optim.SGD(params, lr=lr, momentum=momentum,
                         weight_decay=weight_decay)

    betas = (float(config.get('beta1', 0.9)), float(config.get('beta2', 0.999)))
    eps = float(config.get('eps', 1e-8))
    cls = optim.AdamW if name == 'adamw' else optim.Adam
    # An explicit learning_rate carried over from an SGD config will be far too
    # large here, and that is easy to miss, so say so.
    if 'learning_rate' in config and lr > 2e-3:
        log.warning(f'xvunet optimizer: {name} with learning_rate={lr} is very '
                    'large for an Adam-family optimizer (typical 1e-4..1e-3); '
                    'is this a rate that was tuned for SGD?')
    log.info(f'xvunet optimizer: {cls.__name__} lr={lr} betas={betas} '
             f'eps={eps} weight_decay={weight_decay}')
    return cls(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
