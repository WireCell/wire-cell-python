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


def Optimizer(params, config=None):
    config = config or dict()
    lr = float(config.get('learning_rate', 0.1))
    momentum = float(config.get('momentum', 0.9))
    weight_decay = float(config.get('weight_decay', 0.0005))
    # skip frozen parameters (e.g. warm-started trunks with freeze_unets=True)
    params = [p for p in params if p.requires_grad]
    return optim.SGD(params, lr=lr, momentum=momentum,
                     weight_decay=weight_decay)
