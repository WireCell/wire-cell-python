#!/usr/bin/env python
from torch import optim

## The "app" API
from .model import Network
from .data import Dataset
from wirecell.dnn.train import Classifier as Trainer
# from torch.nn import BCELoss as Criterion
from torch.nn import BCEWithLogitsLoss as Criterion


def Optimizer(params):
    return optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=0.0005)



