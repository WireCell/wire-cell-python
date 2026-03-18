#!/usr/bin/env python
from torch import optim

## The "app" API
from .model import Network
from .data import Dataset
from wirecell.dnn.train import Classifier as Trainer
# from torch.nn import MSELoss as Criterion
from torch.nn.functional import binary_cross_entropy, mse_loss

def Optimizer(params):
    return optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=0.0005)

def Criterion():
    def hurdle_loss(pred, target):
        threshold = .005
        reg_weight = 1.
        target_mask = (target > threshold)
        cls_loss = binary_cross_entropy(pred[:, 0].unsqueeze(1), target_mask.float())

        if target_mask.any():
            reg_loss = mse_loss(pred[:, 1].unsqueeze(1)[target_mask], target[target_mask])
        else:
            reg_loss = 0.
        return cls_loss + reg_weight * reg_loss
    return hurdle_loss
