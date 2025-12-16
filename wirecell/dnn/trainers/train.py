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
from torch import optim, no_grad, float16, bfloat16, float32, autocast, amp, any, save, zeros_like, zeros, cat
import torch.nn as nn
import torch.cuda.memory as memory
import torch.cuda as cuda

def dump(name, data):
    # print(f'{name:20s}: {data.shape} {data.dtype} {data.device}')
    return

class Classifier:
    def __init__(self, net, optimizer, criterion = nn.BCELoss(), device='cpu'):
        net.to(device)
        self._device = device
        self.net = net              # model
        self.optimizer = optimizer
        self.criterion = criterion.to(self._device  )
        self.use_amp = True
        self.scaler = amp.GradScaler(device, enabled=self.use_amp)
        self.save_iter = 0
        self.do_save = False


    def loss(self, features, labels):
        with autocast(
                self._device,
                dtype=bfloat16,
                enabled=self.use_amp):
            features = features.to(self._device)
            dump('features', features)
            labels = labels.to(self._device)
            dump('labels', labels)

            prediction = self.net(features)
            labels = self.net.make_label_nodes_full(labels)
            dump('prediction', prediction)


        # print('Pred:', prediction)
        with no_grad():
            # s = nn.Sigmoid()
            # sigpred = s(prediction)
            # print('Pred Sigmoid:', sigpred)
            if self.do_save:
                save((prediction[0] if type(prediction) == tuple else prediction), f'eval_out_{self.save_iter}.pt')
                save((labels[0] if type(labels) == tuple else labels), f'eval_labels_{self.save_iter}.pt')
                save((features[0] if type(features) == tuple else features), f'eval_input_{self.save_iter}.pt')
                self.save_iter += 1

        # print('Labels:', labels)
        # print('Any in Labels:', any(labels))
        # print('Labels shape:', labels.shape)
        # print('Prediction shape:', prediction.shape)
        # print('lables dtype', labels.dtype)
        # print('prediction dtype', prediction.dtype)
        prediction = tuple(p.to(self._device).to(float32) for p in prediction)
        labels = tuple(p.to(self._device).to(float32) for p in labels)
        loss = self.criterion(prediction, labels)
        return loss

    def evaluate(self, data):
        print("EVALUATING")
        losses = list()
        self.do_save = True
        if cuda.is_available():
            memory._record_memory_history(enabled=True)

                
        with no_grad():
            snapshot_at = 5
            for ie, (features, labels) in enumerate(data):
                loss = self.loss(features, labels)
                

                loss = loss.item()
                print('Eval Loss:', loss)
                losses.append(loss)
                if snapshot_at == ie and cuda.is_available():
                    memory._dump_snapshot(f"eval.pickle")
                    memory._record_memory_history(enabled=False)
                    print('Saved eval snapshot')
        self.do_save = False
        return losses


    def epoch(self, data, retain_graph=False):
        '''
        Train over the batches of the data, return list of losses at each batch.
        '''
        self.net.train()
        epoch_losses = list()
        snapshot_mem = True
        snapshot_at = 1
        for ie, (features, labels) in enumerate(data):

            # with autocast(
            #     self._device,
            #     dtype=bfloat16,
            #     enabled=self.use_amp):
            #     loss = self.loss(features, labels)
            loss = self.loss(features, labels)

            # if snapshot_mem:
            #     # try:
            #     memory._dump_snapshot(f"forward.pickle")
            #     print('Saved foward snapshot')
            #     # except Exception as e:
            #         # print(f"Failed to capture memory snapshot {e}")
            
            # loss.backward(retain_graph=retain_graph)
            # self.scaler.scale(loss.to('cuda:0')).backward()
            self.scaler.scale(loss).backward(retain_graph=retain_graph)
            
            if snapshot_mem and snapshot_at == ie and cuda.is_available():
                # try:
                memory._dump_snapshot(f"backward.pickle")
                print('Saved backward snapshot')
                snapshot_mem = False
                memory._record_memory_history(enabled=None)
            
            # self.optimizer.step()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            loss = loss.item()
            print('Loss:', loss)
            epoch_losses.append(loss)

        return epoch_losses

class Looper:
    def __init__(self, net, optimizer, criterion = nn.BCELoss(), device='cpu'):
        net.to(device)
        self._device = device
        self.net = net              # model
        self.optimizer = optimizer
        self.criterion = criterion
        self.use_amp = True
        self.scaler = amp.GradScaler(device, enabled=self.use_amp)
        self.save_iter = 0
        self.do_loop_eval = True

    def loss(self, features, labels):
        with autocast(
                self._device,
                dtype=bfloat16,
                enabled=self.use_amp):
            features = features.to(self._device)
            dump('features', features)
            labels = labels.to(self._device)
            dump('labels', labels)

            prediction = self.net(features)
            dump('prediction', prediction)
            save(prediction, f'eval_out_{self.save_iter}.pt')

            print('Labels:', labels.shape)
            label_nodes = self.net.make_label_nodes_full(labels)
            print('Labels:', label_nodes.shape)
            print('Node pred:', prediction.shape)
        node_norm = 1./(prediction.size(-2)*labels.size(-1))
        loss = self.criterion(prediction.to(float32), label_nodes.to(float32)) #*node_norm
        
        return loss

    def evaluate(self, data):
        losses = list()
        with no_grad():
            for features, labels in data:
                loss = self.loop_loss(features, labels, training=False, save_pred=True) if self.do_loop_eval else self.loss(features, labels).item()
                save(labels, f'eval_labels_{self.save_iter}.pt')
                save(features, f'eval_input_{self.save_iter}.pt')
                self.save_iter += 1
                losses.append(loss)
        return losses

    def loop_loss(self, features, labels, loss_window=1, training=False, save_pred=False):
        with autocast(
            self._device,
            dtype=bfloat16,
            enabled=self.use_amp):

            #Add if needed
            features = features.to(self._device)
            labels = labels.to(self._device)

            outA, outA_meta = self.net.A(features)

        nregions = outA_meta['nregions']
        
        total_loss_val = 0.0
        total_loss_tensor = 0.0

        nloss_windows = int(nregions/loss_window)
        norm = 1./labels.size(-1)
        print('Norm:', norm, 1./norm)
        if save_pred:
            prediction = []

        for iloss in range(nloss_windows):
            print('Loss window:', iloss)
            with autocast(
                self._device,
                dtype=bfloat16,
                enabled=self.use_amp):
                start = iloss*loss_window
                end = start + loss_window
                label_window = labels[..., start:end]
                outB_i = []
                nodes_outB_i = []
                for t in range(loss_window):
                    i = iloss*loss_window + t
                    if i == nregions: break
                    res = self.net.B(outA, outA_meta, i)
                    outB_i.append(res)

                outB_i = tuple(
                    cat([b[i] for b in outB_i], dim=-1).to(self._device).to(float32)
                    for i in range(len(outB_i[0]))
                )
                if save_pred:
                    prediction.append(outB_i[0])

                label_nodes = self.net.make_label_nodes_full(label_window) #.permute(2,1,0)
                label_nodes = tuple(n.to(float32) for n in label_nodes)
            loss_i = self.criterion(outB_i, label_nodes, do_norm=True)*norm

            total_loss_val += loss_i.item()
            if training: self.scaler.scale(loss_i).backward(retain_graph=(i < (nregions-1)))

        if save_pred:
            save(cat(prediction, dim=-1), f'eval_out_{self.save_iter}.pt')
            # self.save_iter += 1
        return total_loss_val
    def epoch(self, data):
        '''
        Train over the batches of the data, return list of losses at each batch.
        '''
        self.net.train()

        epoch_losses = list()
        snapshot_at = 1
        snapshot_mem = True
        loss_window = 50
        for ie, (features, labels) in enumerate(data):

            total_loss_val = self.loop_loss(features, labels, loss_window=loss_window, training=True)
            print('Loss:', total_loss_val)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if snapshot_mem and snapshot_at == ie and cuda.is_available():
                memory._dump_snapshot(f"backward_loop.pickle")
                print('Saved backward snapshot')
                snapshot_mem = False
                memory._record_memory_history(enabled=None)
            epoch_losses.append(total_loss_val)

        return epoch_losses
