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


class Looper:
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


    def epoch(self, data):
        '''
        Train over the batches of the data, return list of losses at each batch.
        '''
        self.net.train()

        epoch_losses = list()
        for features, labels in data:

            #Add if needed
            # features = features.to(self._device)
            # labels = labels.to(self._device)

            outA, outA_meta = self.net.A(features)

            print('all_crossings:', outA['all_crossings'].shape)
            print('all_neighbors:', outA['all_neighbors'].shape)
            print('edge_attr:', outA['edge_attr'].shape)
            print('labels:', labels.shape)
            nregions = outA_meta['nregions']

            total_loss_val = 0.0
            total_loss_tensor = 0.0
            # for i in range(nregions):
            for i in range(100):
                outB_i = self.net.B(outA, outA_meta, i)
                print('outB_i shape:', outB_i.shape)
                loss_i = self.criterion(outB_i, labels[..., i])
                total_loss_val += loss_i.item()
                total_loss_tensor = total_loss_tensor + loss_i
                
            total_loss_tensor.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()
            print('Total loss:', total_loss_val)
            epoch_losses.append(total_loss_val)

        return epoch_losses
