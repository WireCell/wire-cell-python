#!/usr/bin/env python
'''
DNN training

- network
- features
- labels
- loss / criterion function
- optimizer (network, opt config)

Classifier training

- network.train()
- loop on epochs:
  - loop on training 
    - out = net(features)
    - loss = criterion(out, labels)
    - loss.backward()
    - optimizer.step()

'''
from torch import optim
import torch.nn as nn

class Classifier:
    def __init__(self, net, optclass = optim.SGD, **optkwds):
        self.net = net              # model
        self.optimizer = optclass(net.parameters(), **optkwds)

    def epoch(self, data, criterion=nn.BCELoss()):
        '''
        One train over the batches of the data, return list of losses at each batch.
        '''
        epoch_losses = list()
        for features, labels in data:
            self.optimizer.zero_grad()

            prediction = self.net(src)
            loss = criterion(prediction, labels)
            loss.backward()
            self.optimizer.step()
            epoch_losses.append( loss.item() )
        return epoch_losses

