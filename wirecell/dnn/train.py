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
                save(prediction, f'eval_out_{self.save_iter}.pt')
                save(labels, f'eval_labels_{self.save_iter}.pt')
                save(features, f'eval_input_{self.save_iter}.pt')
                self.save_iter += 1

        # print('Labels:', labels)
        # print('Any in Labels:', any(labels))
        print('Labels shape:', labels.shape)
        print('Prediction shape:', prediction.shape)
        print('lables dtype', labels.dtype)
        print('prediction dtype', prediction.dtype)
        loss = self.criterion(prediction.to(self._device).to(float32), labels.to(float32))
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
        self.save_iter = 0
    def loss(self, features, labels):

        features = features.to(self._device)
        dump('features', features)
        labels = labels.to(self._device)
        dump('labels', labels)

        prediction = self.net(features)
        dump('prediction', prediction)
        # s = nn.Sigmoid()
        # sigpred = s(prediction)
        # print('Pred Sigmoid:', sigpred)
        save(prediction[0], f'eval_out_{self.save_iter}.pt')

        loss = self.criterion(prediction[0], labels)

        print('Labels:', labels.shape)
        label_nodes = self.net.make_label_nodes(labels)
        print('Labels:', label_nodes.shape)
        print('Node pred:', prediction[1].shape)
        node_norm = 1./(prediction[1].size(-2)*labels.size(-1))
        # print('Node norm:', node_norm)
        loss += self.criterion(prediction[1], label_nodes) #*node_norm
        
        return loss

    def evaluate(self, data):
        losses = list()
        with no_grad():
            for features, labels in data:
                # outA, outA_meta = self.A(features)
                # nregions = outA_meta['nregions']
                # out = torch.cat(
                #     [self.B(outA, outA_meta, i) for i in range(nregions)],
                #     dim=-1
                # )
                loss = self.loss(features, labels)
                save(labels, f'eval_labels_{self.save_iter}.pt')
                save(features, f'eval_input_{self.save_iter}.pt')
                self.save_iter += 1
                loss = loss.item()
                losses.append(loss)
        return losses


    def epoch(self, data):
        '''
        Train over the batches of the data, return list of losses at each batch.
        '''
        self.net.train()

        epoch_losses = list()
        snapshot_at = 1
        snapshot_mem = True
        # if not snapshot_mem:
        #     memory._record_memory_history(enabled=False)
        loss_window = 3
        for ie, (features, labels) in enumerate(data):

            #Add if needed
            features = features.to(self._device)
            labels = labels.to(self._device)

            outA, outA_meta = self.net.A(features)

            # print('all_crossings:', outA['all_crossings'].shape)
            # print('all_neighbors:', outA['all_neighbors'].shape)
            # print('edge_attr:', outA['edge_attr'].shape)
            # print('labels:', labels.shape)
            nregions = outA_meta['nregions']
            # nregions=100
            
            total_loss_val = 0.0
            total_loss_tensor = 0.0

            nloss_windows = int(nregions/loss_window)
            norm = 1./(labels.size(-1)*labels.size(-2))
            print('Norm:', norm, 1./norm)
            
            for iloss in range(nloss_windows):
                # print('Loss window:', iloss)
                start = iloss*loss_window
                end = start + loss_window
                label_window = labels[..., start:end]
                outB_i = zeros_like(label_window)
                nodes_outB_i = []
                for t in range(loss_window):
                    i = iloss*loss_window + t
                    # print('\t', t, i)
                    if i == nregions: break
                    res = self.net.B(outA, outA_meta, i)
                    outB_i[..., t] = res[0]
                    # print('node out', res[1].shape)
                    nodes_outB_i.append(res[1])
                    
                # print(outB_i, label_window)
                loss_i = self.criterion(outB_i, label_window)*norm
                # print('loss_i', loss_i)
                nodes_outB_i = cat(nodes_outB_i)
                # print('Nodes out', nodes_outB_i.shape)
                
                label_nodes = self.net.make_label_nodes(label_window).permute(2,1,0)
                # print('Labels:', label_nodes.shape)
                node_norm = 1./(nodes_outB_i.size(-2)*labels.size(-1))
                # print('Node norm:', node_norm)
                loss_i += self.criterion(nodes_outB_i, label_nodes)*node_norm
                # print(loss_i_nodes)

                total_loss_val += loss_i.item()
                loss_i.backward(retain_graph=(i < (nregions-1)))

            print('Loss:', total_loss_val)

            # for i in range(nregions):
            #     print('Region', i)
            #     outB_i = self.net.B(outA, outA_meta, i)
            #     # print('outB_i shape:', outB_i.shape)
            #     loss_i = self.criterion(outB_i, labels[..., i])
            #     total_loss_val += loss_i.item()
            #     # total_loss_tensor = total_loss_tensor + loss_i
                
            #     # total_loss_tensor.backward(retain_graph=(i < (nregions-1))) 
            #     loss_i.backward(retain_graph=(i < (nregions-1))) 

            self.optimizer.step()
            self.optimizer.zero_grad()
            if snapshot_mem and snapshot_at == ie and cuda.is_available():
                # try:
                memory._dump_snapshot(f"backward_loop.pickle")
                print('Saved backward snapshot')
                snapshot_mem = False
                memory._record_memory_history(enabled=None)
            # print('Total loss:', total_loss_val)
            epoch_losses.append(total_loss_val)

        return epoch_losses

class Looper2:
    def __init__(self, net, optimizer, criterion = nn.BCELoss(), device='cpu'):
        net.to(device)
        self._device = device
        self.net = net              # model
        self.optimizer = optimizer
        self.criterion = criterion
        self.use_amp = True
        self.scaler = amp.GradScaler(device, enabled=self.use_amp)
        self.save_iter = 0
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
            # s = nn.Sigmoid()
            # sigpred = s(prediction)
            # print('Pred Sigmoid:', sigpred)
            save(prediction, f'eval_out_{self.save_iter}.pt')

            # loss = self.criterion(prediction[0], labels)

            print('Labels:', labels.shape)
            label_nodes = self.net.make_label_nodes_full(labels)
            print('Labels:', label_nodes.shape)
            print('Node pred:', prediction.shape)
        node_norm = 1./(prediction.size(-2)*labels.size(-1))
        # print('Node norm:', node_norm)
        loss = self.criterion(prediction.to(float32), label_nodes.to(float32)) #*node_norm
        
        return loss

    def evaluate(self, data):
        losses = list()
        with no_grad():
            for features, labels in data:
                # outA, outA_meta = self.A(features)
                # nregions = outA_meta['nregions']
                # out = torch.cat(
                #     [self.B(outA, outA_meta, i) for i in range(nregions)],
                #     dim=-1
                # )
                loss = self.loss(features, labels)
                save(labels, f'eval_labels_{self.save_iter}.pt')
                save(features, f'eval_input_{self.save_iter}.pt')
                self.save_iter += 1
                loss = loss.item()
                losses.append(loss)
        return losses


    def epoch(self, data):
        '''
        Train over the batches of the data, return list of losses at each batch.
        '''
        self.net.train()

        epoch_losses = list()
        snapshot_at = 1
        snapshot_mem = True
        # if not snapshot_mem:
        #     memory._record_memory_history(enabled=False)
        loss_window = 150
        for ie, (features, labels) in enumerate(data):

            with autocast(
                self._device,
                dtype=bfloat16,
                enabled=self.use_amp):

                #Add if needed
                features = features.to(self._device)
                labels = labels.to(self._device)

                outA, outA_meta = self.net.A(features)
                # print('outA', outA.dtype)
                # print('Called A, mem:', cuda.memory_allocated(0) / (1024**2))

            # print('all_crossings:', outA['all_crossings'].shape)
            # print('all_neighbors:', outA['all_neighbors'].shape)
            # print('edge_attr:', outA['edge_attr'].shape)
            # print('labels:', labels.shape)
            nregions = outA_meta['nregions']
            # nregions=100
            
            total_loss_val = 0.0
            total_loss_tensor = 0.0

            nloss_windows = int(nregions/loss_window)
            norm = 1./(labels.size(-1)*labels.size(-2))
            print('Norm:', norm, 1./norm)
            
            for iloss in range(nloss_windows):
                print('Loss window:', iloss)
                with autocast(
                    self._device,
                    dtype=bfloat16,
                    enabled=self.use_amp):
                    start = iloss*loss_window
                    end = start + loss_window
                    label_window = labels[..., start:end]
                    # outB_i = zeros_like(label_window)
                    outB_i = []
                    nodes_outB_i = []
                    for t in range(loss_window):
                        i = iloss*loss_window + t
                        if i == nregions: break
                        res = self.net.B(outA, outA_meta, i)
                        # outB_i[..., t] = res[0]
                        outB_i.append(res)
                        # nodes_outB_i.append(res[1])
                    outB_i = cat(outB_i, dim=-1)
                    # loss_i = self.criterion(outB_i, label_window)*norm
                    # nodes_outB_i = cat(nodes_outB_i)
                    # print('Called B, mem:', cuda.memory_allocated(0) / (1024**2))
                    label_nodes = self.net.make_label_nodes_full(label_window) #.permute(2,1,0)
#                 print('Made Label nodes, mem:', cuda.memory_allocated(0) / (1024**2))
                # print('LABEL NODES', label_nodes.shape)
                # print('OUTB', outB_i.shape)
                # node_norm = 1./(outB_i.size(-2)*labels.size(-1))
                loss_i = self.criterion(outB_i.to(self._device).to(float32), label_nodes.to(float32))*norm

                total_loss_val += loss_i.item()
#                 print('Called loss, mem:', cuda.memory_allocated(0) / (1024**2))
                # loss_i.backward(retain_graph=(i < (nregions-1)))
                self.scaler.scale(loss_i).backward(retain_graph=(i < (nregions-1)))
#                 print('Called backward, mem:', cuda.memory_allocated(0) / (1024**2))

            print('Loss:', total_loss_val)

            # for i in range(nregions):
            #     print('Region', i)
            #     outB_i = self.net.B(outA, outA_meta, i)
            #     # print('outB_i shape:', outB_i.shape)
            #     loss_i = self.criterion(outB_i, labels[..., i])
            #     total_loss_val += loss_i.item()
            #     # total_loss_tensor = total_loss_tensor + loss_i
                
            #     # total_loss_tensor.backward(retain_graph=(i < (nregions-1))) 
            #     loss_i.backward(retain_graph=(i < (nregions-1))) 

            # self.optimizer.step()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if snapshot_mem and snapshot_at == ie and cuda.is_available():
                # try:
                memory._dump_snapshot(f"backward_loop.pickle")
                print('Saved backward snapshot')
                snapshot_mem = False
                memory._record_memory_history(enabled=None)
            # print('Total loss:', total_loss_val)
            epoch_losses.append(total_loss_val)

        return epoch_losses
