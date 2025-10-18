import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from wirecell.dnn.models.unet import UNet
from wirecell.raygrid.coordinates import Coordinates
from wirecell.raygrid import crossover as xover
from wirecell.util.wires import schema, persist

import math
class Network(nn.Module):



    def __init__(self, wires_file='protodunevd-wires-larsoft-v3.json.bz2', nfeatures=4, time_window=3, detector=0):
        super().__init__()
        self.nfeat_post_unet=nfeatures
        
        ##Set up the UNets
        self.unets = nn.ModuleList([
                UNet(n_channels=2, n_classes=nfeatures,
                     batch_norm=True, bilinear=True, padding=True)
                for i in range(3)
        ])
        self.time_window = time_window

        # Get the schema from the file
        store = persist.load(wires_file)

        face_ids = [0, 1]
        self.faces = [store.faces[f] for f in face_ids]
        face_to_planes = {}
        for i, face in enumerate(self.faces):
            face_to_planes[i] = [store.planes[p] for p in face.planes]


        #Do this face-by-face of the detector
        #Treat disconnected planes separately

        #For now -- NEED TO REPLACE WITH SOMETHING IN DATA/TRANSFORM OR SOMETHING UPSTREAM?
        chanmap_npy = np.load('chanmap_1536.npy')

        #maps from chanident to index in input arrays
        chanmap = {c:i for i, c in chanmap_npy}
        # for i, c in chanmap_npy: print(i,c)


        #Build the map to go between wire segments & channels 
        self.face_plane_wires_channels = {}
        for i, face in enumerate(self.faces):
            for j in face.planes:
                plane = store.planes[j]
                wire_chans = []
                for wi in plane.wires:
                    wire = store.wires[wi]
                    wire_chans.append([wire.ident, chanmap[wire.channel]])
                self.face_plane_wires_channels[(i,j)] = torch.tensor(wire_chans, dtype=int)
        

        # face_to_plane_to_nwires = {
        #     i:[len(store.planes[p].wires) for j,p in enumerate(self.faces[i].planes)] for i in face_ids
        # }
        # print(face_to_plane_to_nwires)
        
        self.nwires_0 = [len(store.planes[i].wires) for i in store.faces[0].planes]
        self.nwires_1 = [len(store.planes[i].wires) for i in store.faces[1].planes]

        self.coords_face0 = xover.coords_from_schema(store, 0)
        self.coords_face1 = xover.coords_from_schema(store, 1)
        
        self.good_indices_0_01 = xover.get_good_crossers(self.coords_face0, 0, 1, self.nwires_0)
        self.good_indices_0_12 = xover.get_good_crossers(self.coords_face0, 1, 2, self.nwires_0)
        self.good_indices_0_20 = xover.get_good_crossers(self.coords_face0, 2, 0, self.nwires_0)

        self.ray_crossings_0_01 = self.coords_face0.ray_crossing(0, self.good_indices_0_01[:,0], 1, self.good_indices_0_01[:,1])
        self.ray_crossings_0_12 = self.coords_face0.ray_crossing(1, self.good_indices_0_12[:,0], 2, self.good_indices_0_12[:,1])
        self.ray_crossings_0_20 = self.coords_face0.ray_crossing(2, self.good_indices_0_20[:,0], 0, self.good_indices_0_20[:,1])
        
        self.nchans = [476, 476, 292, 292]

    def forward(self, x):
        '''
        Input data is assumed to be of shape (nbatch, nfeatures, nchannels, nticks)
        '''
        input_shape = x.shape
        nbatches = x.shape[0]
        nticks = x.shape[-1]

        the_device = x.device
        print(x.shape)
        # xs = [
        #     x[:, :, (0 if i == 0 else sum(self.nchans[:i])):sum(self.nchans[:i+1]), :]
        #     for i, nc in enumerate(self.nchans)
        # ]
        # for x in xs: print(x.shape)

        # #Pass through the unets
        # xs = [
        #     self.unets[(i if i < 3 else 2)](xs[i]) for i in range(len(xs))
        # ]

        # print('passed through unets')
        # for x in xs: print(x.shape)

        # #Cat to get into global channel number shape
        # x = torch.cat(xs, dim=2)

        print(x.shape)

        n_feat_base = x.shape[1]
        
        #For ease
        x = x.permute(0,1,3,2)

        #Convert from channels to wires (values duped for common elec chan)
        #Also expand features to include 'meta' features i.e. wire seg number, elec channel
        n_feat_wire = 2
        new_shape = (x.shape[0], n_feat_base+n_feat_wire, x.shape[2], )
        as_wires_f0_p0 = torch.zeros(new_shape + (len(self.face_plane_wires_channels[0,0]),))
        as_wires_f0_p0[:, :2, :, self.face_plane_wires_channels[0,0][:,0]] = x[..., self.face_plane_wires_channels[0,0][:,1]]

        as_wires_f0_p1 = torch.zeros(new_shape + (len(self.face_plane_wires_channels[0,1]),))
        as_wires_f0_p1[:, :2, :, self.face_plane_wires_channels[0,1][:,0]] = x[..., self.face_plane_wires_channels[0,1][:,1]]
        
        as_wires_f0_p2 = torch.zeros(new_shape + (len(self.face_plane_wires_channels[0,2]),))
        as_wires_f0_p2[:, :2, :, self.face_plane_wires_channels[0,2][:,0]] = x[..., self.face_plane_wires_channels[0,2][:,1]]

        print(as_wires_f0_p0.shape)
        print(as_wires_f0_p1.shape)
        print(as_wires_f0_p2.shape)

        #Put features in last dim
        as_wires_f0_p0 = as_wires_f0_p0.permute(0, 2, 3, 1)
        as_wires_f0_p1 = as_wires_f0_p1.permute(0, 2, 3, 1)
        as_wires_f0_p2 = as_wires_f0_p2.permute(0, 2, 3, 1)

        print(as_wires_f0_p0.shape)
        print(as_wires_f0_p1.shape)
        print(as_wires_f0_p2.shape)

        #Wire segment number
        as_wires_f0_p0[..., n_feat_base] = self.face_plane_wires_channels[0,0][:,0]
        as_wires_f0_p1[..., n_feat_base] = self.face_plane_wires_channels[0,1][:,0]
        as_wires_f0_p2[..., n_feat_base] = self.face_plane_wires_channels[0,2][:,0]

        #elec chan number
        as_wires_f0_p0[..., n_feat_base + 1] = self.face_plane_wires_channels[0,0][:,1]
        as_wires_f0_p1[..., n_feat_base + 1] = self.face_plane_wires_channels[0,1][:,1]
        as_wires_f0_p2[..., n_feat_base + 1] = self.face_plane_wires_channels[0,2][:,1]

        #Could add more things: i.e. channel RMS over readout window.
        #Worth some thought and tests
        
        #Now set up our 2-channel crossings -- these will be our GNN nodes
        print('Crossers 01:', self.good_indices_0_01.shape)
        crossings_01 = torch.cat([
           as_wires_f0_p0[:, :, self.good_indices_0_01[:,0], :],
           as_wires_f0_p1[:, :, self.good_indices_0_01[:,1], :],
            self.ray_crossings_0_01.view(1, 1, -1, 2).repeat(nbatches, nticks, 1, 1), #locations of crossings
        ], dim=-1)

        print(crossings_01.shape)
        # torch.save(crossings_01[:,:,:,(2,6)], 'crossings_01.pt')
        # torch.save(self.good_indices_0_01, 'good_indices_01.pt')

        crossings_12 = torch.cat([
           as_wires_f0_p1[:, :, self.good_indices_0_12[:,0], :],
           as_wires_f0_p2[:, :, self.good_indices_0_12[:,1], :],
           self.ray_crossings_0_12.view(1, 1, -1, 2).repeat(nbatches, nticks, 1, 1), #locations of crossings
        ], dim=-1)

        crossings_20 = torch.cat([
           as_wires_f0_p2[:, :, self.good_indices_0_20[:,0], :],
           as_wires_f0_p0[:, :, self.good_indices_0_20[:,1], :],
           self.ray_crossings_0_20.view(1, 1, -1, 2).repeat(nbatches, nticks, 1, 1), #locations of crossings
        ], dim=-1)

        # time.sleep(10)
        return 0
        padded = F.pad(x, [(self.time_window - 1)/2])
        print('Padded', padded.shape)

        #Is this hacky -- with the clone specifically?
        windowed = torch.zeros_like(x).unsqueeze(-1).expand([-1]*len(x.shape) + [3]).clone()

        #Iterate through each point in time and replicate into the expanded tensor        
        for i in range(x.shape[-1]):
            windowed[..., i,:] = padded[..., i:i+self.time_window]

        #The nodes are 2-wire crossings for each point in time
        

        print(x.shape)

        nbatches = x.shape[0]
        nfeatures_in = x.shape[1]
        nticks = x.shape[2]
        nchans = x.shape[3]

        return 1

