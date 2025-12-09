import numpy as np
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch_scatter
from torch_geometric.data import Data
from torch_geometric.nn import GAT, GCN
from wirecell.dnn.models.unet import UNet
from wirecell.raygrid.coordinates import Coordinates
from wirecell.raygrid import crossover as xover
from wirecell.util.wires import schema, persist
import torch.utils.checkpoint as checkpoint

import math

class Network(nn.Module):
    
    def determine_mlp_n_in(self):
        if self.do_call_first_unets:
            mlp_n_in = 3*self.first_unet_n_out
        else:
            mlp_n_in = 3*self.n_input_features
        return mlp_n_in

    def determine_unets2_n_in(self):
        if self.do_call_mp and self.special_style == 'mlp':
            if self.do_call_first_unets:
                return self.first_unet_n_out + self.mlp_n_out
            else:
                return self.n_input_features + self.mlp_n_out
        elif self.do_call_mp and self.special_style == 'threshold':
            if self.do_call_first_unets:
                return 4*self.first_unet_n_out #Should this be 5?
            else:
                return 4*self.n_input_features #Should this be 5?
        elif self.do_call_mp and self.special_style == 'feedtrough':
            if self.do_call_first_unets:
                return 4*self.first_unet_n_out
            else:
                return 4*self.n_input_features
        else:
            return self.first_unet_n_out
            return self.n_input_features

    # @torch.autocast('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16) #TODO -- Fix this
    def __init__(
            self,
 
            # wires_file='protodunevd-wires-larsoft-v3.json.bz2',
            # chanmap_file='chanmap_1536.npy',
            # nchans=[476, 476, 292, 292],
            # det_type='vd',
            # cells_file=None,
 
            wires_file='protodunehd-wires-larsoft-v1.json.bz2',
            chanmap_file=2560,
            nchans=[800, 800, 480, 480],
            det_type='hd',
            cells_file='pdhd_cells.pt',

            n_unet_features=4,
            # time_window=3,
            checkpoint=False,
            n_feat_wire = 0,
            detector=0,
            n_input_features=1,
            # skip_unets=True,
            # skip_GNN=False,
            # one_side=False,
            # out_channels=16,
            # use_cells=True,
            # fixed_neighbors=True,
            #gcn=False, #Currently not working
        ):
        super().__init__()
        self.nfeat_post_unet=n_unet_features
        self.n_feat_wire = n_feat_wire
        self.checkpoint=checkpoint
        self.n_input_features=n_input_features
        # self.skip_unets=skip_unets
        ##Set up the UNets
        
        self.do_call_first_unets = True
        self.do_call_second_unets = True
        self.do_call_mp = True
        self.split_gpu = True
        self.do_call_special = True
        self.special_style = 'mlp' #mlp, threshold, or feedthrough
        self.good_specials = ['mlp', 'threshold', 'feedthrough']
        if self.special_style not in self.good_specials:
            raise Exception(f'Unknown Special Style {self.special_style}. Can only select one of {(", ").join(self.good_specials)}')
        self.mlp_n_out = 2 #Only used if special_style set to mlp
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.xview_activation = self.relu

        self.network_style = 'U-MP-U'
        if self.network_style is not None:
            if self.network_style == 'U-MP-U':
                self.do_call_first_unets = True
                self.do_call_second_unets = True
                self.do_call_special = True
                self.do_call_mp = True
            elif self.network_style == 'MP-U':
                self.do_call_first_unets = False
                self.do_call_second_unets = True
                self.do_call_special = True
                self.do_call_mp = True
            elif self.network_style == 'U':
                self.do_call_special = False
                self.do_call_first_unets = False
                self.do_call_second_unets = True
                self.do_call_mp = False
            else:
                raise Exception('Uknown network style', self.network_style)
            
            print('Chose network style', self.network_style)


        
        self.first_unet_n_out=1
        if self.do_call_first_unets:
            self.unets = nn.ModuleList([
                    UNet(n_channels=n_input_features, n_classes=self.first_unet_n_out,
                        batch_norm=True, bilinear=True, padding=True)
                    for i in range(3)
            ])

        mlp_n_in = self.determine_mlp_n_in()
        
        if self.special_style == 'feedthrough':
            self.do_call_special=False
        elif self.special_style == 'mlp' and self.do_call_special:
            self.mlp = nn.Sequential(
                nn.Linear(mlp_n_in, 8),
                nn.Linear(8, self.mlp_n_out)
            )
        
        self.unet2_device = 'cuda:1' if (self.split_gpu and self.do_call_second_unets) else 'cuda:0'
        self.second_unet_n_in = self.determine_unets2_n_in()
        if self.do_call_second_unets:
            
            # if self.do_call_special and self.special_style == 'mlp':
            #     self.second_unet_n_in = 1 + self.mlp_n_out
            # elif self.do_call_special and self.special_style == 'threshold':
            #     self.second_unet_n_in = 4*self.first_unet_n_out
            # else:
            #     self.second_unet_n_in = 4

            self.unets2 = nn.ModuleList([
                    UNet(n_channels=self.second_unet_n_in, n_classes=1,
                        batch_norm=True, bilinear=True, padding=True)
                    for i in range(3)
            ])


        with torch.no_grad():

            # self.time_window = time_window

            # Get the schema from the file
            store = persist.load(wires_file)

            face_ids = [0, 1]
            self.faces = [store.faces[f] for f in face_ids]
            face_to_planes = {}
            for i, face in enumerate(self.faces):
                face_to_planes[i] = [store.planes[p] for p in face.planes]


            #Do this face-by-face of the detector
            #Treat disconnected planes separately

            if type(chanmap_file) == str:
                #For now -- NEED TO REPLACE WITH SOMETHING IN DATA/TRANSFORM OR SOMETHING UPSTREAM?
                #This comes from channels_X in one of the input files
                #The index of the array is the 'global' channel number -- index into frame array.
                #The value of the array is the wire.channel -- larsoft channel ID 
                chanmap_npy = np.load('chanmap_1536.npy')

                #maps from chanident to index in input arrays
                chanmap = {c:i for i, c in chanmap_npy}
                # for i, c in chanmap_npy: print(i,c)
            elif type(chanmap_file) == int:
                chanmap = {i:i  for i in range(chanmap_file)}
            else:
                raise Exception(f'Expected either str or int for chanmap_file type but received {type(chanmap_file)}')


            #Build the map to go between wire segments & channels 
            self.face_plane_wires_channels = {}
            for i, face in enumerate(self.faces):
                for jj, j in enumerate(face.planes):
                    plane = store.planes[j]
                    wire_chans = torch.zeros((len(plane.wires), 2), dtype=int)
                    for wi in plane.wires:
                        wire = store.wires[wi]
                        wire_chans[wire.ident, 0] = wire.ident
                        wire_chans[wire.ident, 1] = chanmap[wire.channel]
                    self.face_plane_wires_channels[(i,jj)] = torch.tensor(wire_chans, dtype=int)
            

            # face_to_plane_to_nwires = {
            #     i:[len(store.planes[p].wires) for j,p in enumerate(self.faces[i].planes)] for i in face_ids
            # }
            # print(face_to_plane_to_nwires)
            
            self.nwires_0 = [len(store.planes[i].wires) for i in store.faces[0].planes]
            self.nwires_1 = [len(store.planes[i].wires) for i in store.faces[1].planes]

            self.coords_face0 = xover.coords_from_schema(store, 0)
            self.coords_face1 = xover.coords_from_schema(store, 1, left_handed=(det_type == 'hd'))

            if cells_file == None:
                print('Making cells face 0')
                self.good_indices_0 = xover.make_cells(self.coords_face0, *(self.nwires_0), keep_shape=False)
                print('Done', self.good_indices_0.shape)
                print('Making cells face 1')
                self.good_indices_1 = xover.make_cells(self.coords_face1, *(self.nwires_1), keep_shape=False)
                print('Done', self.good_indices_1.shape)
            else:
                cells_from_file = torch.load(cells_file)
                self.good_indices_0 = cells_from_file['cells_face0']
                self.good_indices_1 = cells_from_file['cells_face1']

            self.nchans = nchans #[476, 476, 292, 292]

            self.save = True
    
    # def fill_window_mp(self, first_wires, second_wires, third_wires, indices, type='mp3'):
    #     w = second_wires*third_wires

    #     if type == 'mp3':
    #         w = w*first_wires
    #     elif type == 'mp2':
    #         w = 1 - w*first_wires
    #     return w
            
    # def fill_window(self, w, nfeat, first_wires, second_wires, third_wires, indices, crossings, merged_crossings):
        
    #     w[..., :(nfeat)] = first_wires[:, :, indices[:,0], :]
    #     w[..., (nfeat):2*(nfeat)] = second_wires[:, :, indices[:,1], :]
    #     w[..., 2*(nfeat):3*(nfeat)] = third_wires[:, :, indices[:,2], :]
    #     start = 3*(nfeat)

    #     if not self.skip_area:
    #         w[..., start] = merged_crossings['areas']
    #         start += 1
        # w[..., start:start+2] = crossings[:, 0].view(1, 1, -1, 2).repeat(w.shape[0], w.shape[1], 1, 1).to(w.device)
        # start += 2
        # w[..., start:start+2] = crossings[:, 1].view(1, 1, -1, 2).repeat(w.shape[0], w.shape[1], 1, 1).to(w.device)
        # start += 2
        # w[..., start:start+2] = crossings[:, 2].view(1, 1, -1, 2).repeat(w.shape[0], w.shape[1], 1, 1).to(w.device)
        # start += 2
        # w[..., start:start+2] = torch.mean(crossings, dim=1).view(1,1,-1,2).repeat(w.shape[0], w.shape[1], 1, 1).to(w.device)
    

    def scatter_to_chans(self, y, nbatches, nchannels, the_device):
        #TODO -- check size of y etc
        temp_out = torch.zeros(nbatches, y[0].shape[1], nchannels, y[0].size(-1)).to(the_device)

        # for k, v in self.face_plane_wires_channels.items():
        #     self.face_plane_wires_channels[k] = v.to(the_device)
        # print('scatter device', the_device)
        # print('Inds devices', self.good_indices_0.device, self.good_indices_1.device)
        # print('input devices', y[0].device, y[1].device)
        # temp_out2 = temp_out.clone()
        # mem0 = torch.cuda.memory_allocated(0) / (1024**2)
        to_scatter = [
            [y[0], self.face_plane_wires_channels[(0,0)], self.good_indices_0[:,0]],
            
            #plane1
            [y[0], self.face_plane_wires_channels[(0,1)], self.good_indices_0[:,1]],
            
            #plane2
            [y[0], self.face_plane_wires_channels[(0,2)], self.good_indices_0[:,2]],

            #plane0
            [y[1], self.face_plane_wires_channels[(1,0)], self.good_indices_1[:,0]],

            #plane1
            [y[1], self.face_plane_wires_channels[(1,1)], self.good_indices_1[:,1]],
            
            #plane2
            [y[1], self.face_plane_wires_channels[(1,2)], self.good_indices_1[:,2]],
        ]
        # mem1 = torch.cuda.memory_allocated(0) / (1024**2)
        # print(f'In scatter {mem0:.2f} {mem1:.2f}')

        for yi, wire_chans, indices in to_scatter:
            # print(yi.shape, indices.shape, wire_chans.shape)
            # print(wire_chans[indices][:,1].unsqueeze(-1).unsqueeze(0).unsqueeze(1).shape)
            # torch_scatter.scatter_add(
            # torch_scatter.scatter_max(
            # torch_scatter.scatter_mean(
            #     yi, indices,
            #     out=temp_out,
            #     dim=1
            # )
            temp_out = temp_out.scatter_reduce(
                2,
                # wire_chans[indices][:,1].unsqueeze(1).unsqueeze(0).repeat(1, 1, yi.shape[-1]),
                wire_chans.to(the_device)[indices.to(the_device)][:,1].unsqueeze(-1).unsqueeze(0).unsqueeze(1).repeat(yi.shape[0], yi.shape[1], 1, yi.shape[-1]),
                yi,
                'amax',
                include_self=True,
            )

        # print(temp_out - temp_out2)
        return temp_out

    def make_wires(self, x, face, plane):
        wire_chans = self.face_plane_wires_channels[(face, plane)].to(x.device)
        # print(wire_chans)
        # print('Input', x.shape[0], len(wire_chans), x.shape[-1])
        wires = torch.zeros((x.shape[0], x.shape[1], len(wire_chans), x.shape[-1])).to(x.device)
        # print('Wires')
        # print(x.shape, wires.shape, torch.max(wire_chans[:, 0]), torch.max(wire_chans[:, 1]))
        wires[..., wire_chans[:, 0], :] = x[..., wire_chans[:,1], :]
        # wires[..., x.shape[-1]] = plane
        # wires[..., x.shape[-1]+1] = face
        return wires

    def split_x(self, x):
        print('Splitting')
        x = [
            x[:, :, (0 if i == 0 else sum(self.nchans[:i])):sum(self.nchans[:i+1]), :]
            for i, nc in enumerate(self.nchans)
        ]
        for xi in x: print(xi.shape)
        return x

    def call_unets(self, x, unets):
        x = self.split_x(x)
        x = [
            (unets[(i if i < 3 else 2)](x[i]) if not self.checkpoint else
            checkpoint.checkpoint(
                unets[(i if i < 3 else 2)],
                x[i]
            )) for i in range(len(x))
        ]
        print('passed through unets')
        for xi in x: print(xi.shape)

        #Cat to get into global channel number shape
        x = torch.cat(x, dim=2)
        return x

    def mp_step(self, x):
        # x = self.sigmoid(x)
        x = self.xview_activation(x)
        x = self.calculate_crossview(x, call_special=self.do_call_special)
        return x

    def forward(self, x):
        '''
        Input data is assumed to be of shape (nbatch, nfeatures, nchannels, nticks)
        '''
        input_shape = x.shape
        nbatches = x.shape[0]
        nticks = x.shape[-1]
        nchannels = x.shape[-2]

        the_device = x.device
        self.good_indices_0 = self.good_indices_0.to(the_device)
        self.good_indices_1 = self.good_indices_1.to(the_device)
        if self.do_call_second_unets:
            if the_device != 'cpu':
                split_device = self.unet2_device
            else: split_device = 'cpu'
            self.unets2 = nn.ModuleList([ui.to(self.unet2_device) for ui in self.unets2])

        # # if not self.skip_unets:
        # if self.save:
        #     torch.save(x, 'input_test.pt')

        if self.do_call_first_unets:
            print('Calling first UNets')
            x = self.call_unets(x, self.unets)

        if self.do_call_mp:
            print('Calling MP')
            x = self.mp_step(x)
        
        if self.do_call_second_unets:
            x = x.to(split_device)
            print('Calling second UNets')
            x = self.call_unets(x, self.unets2)

        x = self.sigmoid(x)
        x = self.calculate_crossview(x) #Currently out of memory on a single 6090 if trying to do this 


        return x.to(the_device)

    def call_mlp(self, x):
        x = self.mlp(
            x.permute(0,2,3,1)
        ).permute(0,3,1,2)
        return x
    
    def call_threshold_like(self, x):
        nfeat = x.shape[1]
        if (nfeat % 3):
            raise Exception(f'Expected number of features to be a multiple of 3, but received {nfeat}')
        nfeat = int(nfeat / 3)
        # print('nfeat', nfeat)
        return torch.cat([
            x[:, 0:nfeat]*x[:, nfeat:2*nfeat]*x[:,2*nfeat:3*nfeat],
            (1-x[:, :nfeat])*x[:, nfeat:2*nfeat]*x[:,2*nfeat:3*nfeat],
            (1-x[:, nfeat:2*nfeat])*x[:,2*nfeat:3*nfeat]*x[:, :nfeat],
            (1-x[:, 2*nfeat:3*nfeat])*x[:,:nfeat]*x[:, nfeat:2*nfeat],
        ], dim=1)
    
    def calculate_crossview(self, input, call_special=False):
        #Now we have to construct MP3 and MP2
        #Go from the values on the channels to wires then make cells
        ##HAVE TO DO THIS IN A LOOP BECAUSE IT'S TOO BIG
        ## SO INSTEAD OF DOING EVERY TIME SAMPLE AT ONCE, DO IT IN A LOOP FILL THE TIME SAMPLE OUTPUT
        ## ONE BY ONE
        nbatches = input.shape[0]
        nchannels = input.shape[-2]
        the_device = input.device
        crossview_chans_mp3 = []
        crossview_chans_mp2 = []
        crossview_chans = []
        self.good_indices_0 = self.good_indices_0.to(the_device)
        self.good_indices_1 = self.good_indices_1.to(the_device)
        for i in range(input.shape[-1]):
            xi = input[..., i].unsqueeze(-1)
            # print(xi.shape)

            #These are called 'as_wires' but are really cells on each face
            as_wires_f0 = torch.cat([
                self.make_wires(xi, 0, 0)[:, :, self.good_indices_0[:, 0]],
                self.make_wires(xi, 0, 1)[:, :, self.good_indices_0[:, 1]],
                self.make_wires(xi, 0, 2)[:, :, self.good_indices_0[:, 2]]
            ], dim=1)

            as_wires_f1 = torch.cat([
                self.make_wires(xi, 1, 0)[:, :, self.good_indices_1[:, 0]],
                self.make_wires(xi, 1, 1)[:, :, self.good_indices_1[:, 1]],
                self.make_wires(xi, 1, 2)[:, :, self.good_indices_1[:, 2]]
            ], dim=1)

            #This mixes the information between the three planes
            if call_special:
                if self.special_style == 'mlp':
                    as_wires_f0 = self.call_mlp(as_wires_f0)
                    as_wires_f1 = self.call_mlp(as_wires_f1)
                elif self.special_style == 'threshold':
                    # print('Calling threshold')
                    as_wires_f0 = self.call_threshold_like(as_wires_f0)
                    as_wires_f1 = self.call_threshold_like(as_wires_f1)

            cross_view_all = self.scatter_to_chans(
                [as_wires_f0, as_wires_f1],
                nbatches, nchannels, the_device
            )

            crossview_chans.append(cross_view_all)
        
        x = torch.cat([
            input,
            torch.cat(crossview_chans, dim=-1)
        ], dim=1)
        return x

    def make_label_nodes(self, labels):
        '''
        We get the labels in a shape of (batch, feat=1, channels, ticks)

        So we need to go from channels view to the wires then nodes view.
        '''

        return self.calculate_crossview(labels)
