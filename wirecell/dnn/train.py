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
from torch import optim, no_grad
import torch.nn as nn

def dump(name, data):
    # print(f'{name:20s}: {data.shape} {data.dtype} {data.device}')
    return

class Classifier:
    def __init__(self, net, optimizer, criterion = nn.BCELoss(), device='cpu'):
        net.to(device)
        self._device = device
        self.net = net              # model
        self.optimizer = optimizer
        self.criterion = criterion

    def loss(self, features, labels):

        features = features.to(self._device)
        dump('features', features)
        labels = labels.to(self._device)
        dump('labels', labels)

        prediction = self.net(features)
        dump('prediction', prediction)

        loss = self.criterion(prediction, labels)
        return loss

    def evaluate(self, data):
        losses = list()
        with no_grad():
            for features, labels in data:
                loss = self.loss(features, labels)
                loss = loss.item()
                losses.append(loss)
        return losses


    def epoch(self, data, retain_graph=False):
        '''
        Train over the batches of the data, return list of losses at each batch.
        '''
        self.net.train()

        epoch_losses = list()
        for features, labels in data:

            loss = self.loss(features, labels)

            loss.backward(retain_graph=retain_graph)
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss = loss.item()
            epoch_losses.append(loss)

        return epoch_losses

