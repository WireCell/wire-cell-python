#!/usr/bin/env python
from torch import optim

## The "app" API
from .model import Network
from .data import Dataset
from wirecell.dnn.train import Classifier as Trainer
from torch.nn import BCEWithLogitsLoss as Criterion


def Optimizer(params, **config):
    lr = float(config.get('learning_rate', 0.1))
    p = float(config.get('momentum', 0.9))
    weight_decay = float(config.get('weight_decay', 0.0005))
    return optim.SGD(params, lr=lr, momentum=p, weight_decay=weight_decay)



