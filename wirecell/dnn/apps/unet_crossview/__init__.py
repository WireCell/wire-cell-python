#!/usr/bin/env python
from torch import optim, tensor, float32 as tf32

## The "app" API
from wirecell.dnn.models.unet_crossview import UNetCrossView as Network
from .data import Dataset
# from wirecell.dnn.trainers.train import Looper as Trainer

from wirecell.dnn.trainers.train import Classifier as Trainer

def Optimizer(params):
    return optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0005)

class Criterion:
    '''Multi-term loss'''
    def __init__(self):
        from torch.nn import BCELoss
        # self.crit = BCELoss(reduction='sum')
        self.crit = BCELoss()
    def to(self, device):
        self.crit = self.crit.to(device)
        return self

    def __call__(self, prediction : tuple, label : tuple, do_norm : bool = False ):
        if len(prediction) != len(label):
            raise Exception(
            'Error! Expected prediction and label to have same size'
            f'but received {len(prediction)} and {len(label)} respectively.')
        loss = 0
        for i, (p, l) in enumerate(zip(prediction, label)):
            norm = 1./(l.shape[-2]) if do_norm else 1.
            loss += self.crit(p, l)*norm
        return loss