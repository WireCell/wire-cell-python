#!/usr/bin/env python


from .model import Network
from .data import Dataset
from .train import Classifier as Trainer


from torch import optim
def Optimizer(params):
    return optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=0.0005)



