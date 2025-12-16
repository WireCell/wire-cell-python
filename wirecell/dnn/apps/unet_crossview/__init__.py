#!/usr/bin/env python
from torch import optim, tensor, float32 as tf32

## The "app" API
from .model import Network
from .data import Dataset
from wirecell.dnn.trainers.train import Looper2 as Trainer
# from wirecell.dnn.train import Classifier as Trainer
# from torch.nn import BCEWithLogitsLoss as Criterion
# from torch.nn import BCELoss as Criterion
# 
# from torch.nn import BCEWithLogitsLoss #as Criterion


def Optimizer(params):
    return optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0005)

# def Criterion():
#     from torch.nn import BCELoss
#     return BCELoss(reduction='sum')

# class Criterion():
#     def __init__(self):
#         from torch.nn import BCEWithLogitsLoss
#         self.crit = BCEWithLogitsLoss(pos_weight=tensor([10,1,1,1], dtype=tf32).unsqueeze(-1).unsqueeze(-1))
#     def to(self, device):
#         self.crit = self.crit.to(device)
#         return self
#     def __call__(self, prediction, target):
        
#         loss = self.crit(prediction, target)
#         return loss

# def Criterion():
    # from torch.nn import BCEWithLogitsLoss
    # return BCEWithLogitsLoss(weight=tensor([1, 10], dtype=tf32))

class Criterion:
    '''Multi-term loss'''
    def __init__(self):
        from torch.nn import BCELoss
        self.crit = BCELoss(reduction='sum')
        # self.crit = BCELoss()
    def to(self, device):
        self.crit = self.crit.to(device)
        return self

    def __call__(self, prediction : tuple, label : tuple, do_norm : bool = False ):
        if len(prediction) != len(label):
            raise Exception(
            'Error! Expected prediction and label to have same size'
            f'but received {len(prediction)} and {len(label)} respectively.')
        loss = 0
        # weights = [1., 2.]
        for i, (p, l) in enumerate(zip(prediction, label)):
            norm = 1./(l.shape[-2]) if do_norm else 1.
            loss += self.crit(p, l)*norm #*weights[i]
        return loss