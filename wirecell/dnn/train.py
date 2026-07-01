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
import torch
from torch import optim, no_grad
import torch.nn as nn

def dump(name, data):
    # print(f'{name:20s}: {data.shape} {data.dtype} {data.device}')
    return

class Classifier:
    def __init__(self, net, optimizer, criterion = nn.BCELoss(), device='cpu', amp=False):
        net.to(device)
        self._device = device
        self.net = net              # model
        self.optimizer = optimizer
        self.criterion = criterion
        # Mixed precision (autocast + GradScaler) is only meaningful on CUDA.
        # When disabled, both autocast and the scaler are pass-throughs, so the
        # code path is behavior-identical to plain fp32 training.
        self._amp = bool(amp) and str(device).startswith('cuda')
        self._scaler = torch.amp.GradScaler('cuda', enabled=self._amp)

    def loss(self, features, labels):

        features = features.to(self._device, non_blocking=True)
        dump('features', features)
        labels = labels.to(self._device, non_blocking=True)
        dump('labels', labels)

        # Only the model forward runs under autocast (the heavy conv work goes
        # fp16).  The prediction leaves autocast in fp16, so cast it back to
        # fp32 and compute the criterion *outside* autocast: BCELoss is
        # autocast-unsafe, and this also keeps the loss numerics in fp32 (the
        # standard AMP recipe).
        with torch.autocast('cuda', enabled=self._amp):
            prediction = self.net(features)
        dump('prediction', prediction)

        loss = self.criterion(prediction.float(), labels)
        return loss

    def evaluate(self, data):
        losses = list()
        # Evaluate in eval mode so BatchNorm uses its running statistics and
        # stops updating its buffers from the validation data.  no_grad() alone
        # does NOT change module mode.  Restore the prior mode afterward.
        was_training = self.net.training
        self.net.eval()
        try:
            with no_grad():
                for features, labels in data:
                    loss = self.loss(features, labels)
                    loss = loss.item()
                    losses.append(loss)
        finally:
            self.net.train(was_training)
        return losses


    def epoch(self, data, retain_graph=False):
        '''
        Train over the batches of the data, return list of losses at each batch.
        '''
        self.net.train()

        epoch_losses = list()
        for features, labels in data:

            loss = self.loss(features, labels)

            self.optimizer.zero_grad(set_to_none=True)
            self._scaler.scale(loss).backward(retain_graph=retain_graph)
            self._scaler.step(self.optimizer)
            self._scaler.update()

            loss = loss.item()
            epoch_losses.append(loss)

        return epoch_losses

