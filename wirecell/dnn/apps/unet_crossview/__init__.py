#!/usr/bin/env python
from torch import optim

## The "app" API
from .model import Network
from .data import Dataset
from wirecell.dnn.train import Classifier as Trainer
from torch.nn import BCELoss as Criterion
# from torch.nn import BCEWithLogitsLoss #as Criterion


def Optimizer(params):
    return optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=0.0005)

# def Criterion():
#     return BCELoss(reduction='sum')
    # return BCEWithLogitsLoss(reduction='sum')

# class Criterion:
#     def __init__(self):
#         self.criterion = BCELoss(reduction='sum')

#     def __call__(self, prediction, loss):
#         '''
#         Prediction and labels come in as shape (batch, feat, channel, tick)
#         '''

        
