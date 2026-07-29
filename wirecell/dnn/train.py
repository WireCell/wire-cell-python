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

from wirecell.dnn import dist

def dump(name, data):
    # print(f'{name:20s}: {data.shape} {data.dtype} {data.device}')
    return


# Autocast dtypes accepted for --amp.  fp16 carries a much narrower exponent
# range than fp32 (max ~65504); bf16 keeps fp32's range at the same speed.  That
# matters for models whose activations can overflow: with xvunet's cross-view
# branch, fp16 drove ~29% of batches to inf logits (silently skipped by the
# GradScaler) once its LayerScale gates opened, while bf16 tracked fp32 to
# ~0.05% at an identical 380 ms/step.
_AMP_DTYPES = dict(float16=torch.float16, fp16=torch.float16, half=torch.float16,
                   bfloat16=torch.bfloat16, bf16=torch.bfloat16)


class Classifier:
    def __init__(self, net, optimizer, criterion = nn.BCELoss(), device='cpu',
                 amp=False, amp_dtype='float16'):
        net.to(device)
        self._device = device
        # Under DDP each process wraps its replica in DistributedDataParallel so
        # gradients are all-reduced across GPUs every backward pass.  The raw
        # `net` reference held by the caller is left untouched (it is used there
        # for checkpoint load/save, keeping state-dict keys free of a `module.`
        # prefix).  Not distributed -> self.net is just the plain module.
        if dist.is_dist():
            from torch.nn.parallel import DistributedDataParallel
            self.net = DistributedDataParallel(net, device_ids=[dist.get_local_rank()])
        else:
            self.net = net          # model
        self.optimizer = optimizer
        self.criterion = criterion
        # Mixed precision (autocast + GradScaler) is only meaningful on CUDA.
        # When disabled, both autocast and the scaler are pass-throughs, so the
        # code path is behavior-identical to plain fp32 training.
        self._amp = bool(amp) and str(device).startswith('cuda')
        key = str(amp_dtype).lower()
        if key not in _AMP_DTYPES:
            raise ValueError(f'unknown amp dtype {amp_dtype!r}, want one of '
                             f'{sorted(set(_AMP_DTYPES))}')
        self._amp_dtype = _AMP_DTYPES[key]
        # Loss scaling exists to lift fp16 gradients out of the subnormal range.
        # bf16 has fp32's exponent, so it neither needs nor benefits from it.
        self._scaler = torch.amp.GradScaler(
            'cuda', enabled=self._amp and self._amp_dtype is torch.float16)

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
        with torch.autocast('cuda', dtype=self._amp_dtype, enabled=self._amp):
            prediction = self.net(features)
        dump('prediction', prediction)

        loss = self.criterion(prediction.float(), labels)
        return loss

    def _global_mean(self, total_loss, total_n):
        '''
        Return the per-sample mean loss.  Under DDP the (sum-of-loss, count)
        pair is all-reduced across ranks first so every process reports the
        same global metric instead of its local shard's mean.
        '''
        if dist.is_dist():
            stats = torch.tensor([total_loss, total_n], dtype=torch.float64,
                                  device=self._device)
            dist.all_reduce_sum(stats)
            total_loss, total_n = stats[0].item(), stats[1].item()
        return total_loss / total_n if total_n else 0.0

    def evaluate(self, data):
        '''
        Evaluate over the batches of data.  Return (mean_loss, per_batch_losses)
        where mean_loss is the per-sample mean weighted by batch size (correct
        for any batch size, including a partial final batch).
        '''
        losses = list()
        total_loss = 0.0
        total_n = 0
        # Evaluate in eval mode so BatchNorm uses its running statistics and
        # stops updating its buffers from the validation data.  no_grad() alone
        # does NOT change module mode.  Restore the prior mode afterward.
        was_training = self.net.training
        self.net.eval()
        try:
            with no_grad():
                for features, labels in data:
                    loss = self.loss(features, labels).item()
                    n = features.shape[0]
                    losses.append(loss)
                    total_loss += loss * n
                    total_n += n
        finally:
            self.net.train(was_training)
        mean_loss = self._global_mean(total_loss, total_n)
        return mean_loss, losses


    def epoch(self, data, retain_graph=False):
        '''
        Train over the batches of the data.  Return (mean_loss, per_batch_losses)
        where mean_loss is the per-sample mean weighted by batch size (correct
        for any batch size, including a partial final batch).
        '''
        self.net.train()

        epoch_losses = list()
        total_loss = 0.0
        total_n = 0
        for features, labels in data:

            loss = self.loss(features, labels)

            self.optimizer.zero_grad(set_to_none=True)
            self._scaler.scale(loss).backward(retain_graph=retain_graph)
            self._scaler.step(self.optimizer)
            self._scaler.update()

            n = features.shape[0]
            loss = loss.item()
            epoch_losses.append(loss)
            total_loss += loss * n
            total_n += n

        mean_loss = self._global_mean(total_loss, total_n)
        return mean_loss, epoch_losses

