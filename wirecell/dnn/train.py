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

def dump(name, data):
    # print(f'{name:20s}: {data.shape} {data.dtype} {data.device}')
    return

class Classifier:
    def __init__(self, net, device='cpu', optclass = optim.SGD, **optkwds):
        net.to(device)
        self._device = device
        self.net = net              # model
        self.optimizer = optclass(net.parameters(), **optkwds)

    def epoch(self, data, criterion=nn.BCELoss(), retain_graph=False):
        '''
        Train over the batches of the data, return list of losses at each batch.
        '''
        self.net.train()

        epoch_losses = list()
        for features, labels in data:

            features = features.to(self._device)
            dump('features', features)
            labels = labels.to(self._device)
            dump('labels', labels)

            prediction = self.net(features)
            dump('prediction', prediction)

            loss = criterion(prediction, labels)

            loss.backward(retain_graph=retain_graph)
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss = loss.item()
            epoch_losses.append(loss)

        return epoch_losses

