import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GAT
from wirecell.dnn.models.unet import UNet
from wirecell.raygrid.coordinates import Coordinates
from wirecell.raygrid import crossover as xover
from wirecell.util.wires import schema, persist

def check_A_in_B(A, B):
    B_map = {
        tuple(row.tolist()): i
        for i, row in enumerate(B)
    }
    A_indices_in_B = []
    A_indices = []
    for i, row_a in enumerate(A):
        row_tuple = tuple(row_a.tolist())
        
        if row_tuple in B_map:
            # Found the row: append the index from B
            A_indices_in_B.append(B_map[row_tuple])
            A_indices.append(i)
    A_indices_in_B_tensor = torch.tensor(A_indices_in_B, dtype=torch.long)
    A_indices_tensor = torch.tensor(A_indices, dtype=torch.long)
    return torch.cat([
        A_indices_tensor.unsqueeze(0),
        A_indices_in_B_tensor.unsqueeze(0),
    ])

def get_nn_from_plane_pair(indices, n_nearest=3):
    expanded = indices.unsqueeze(0).repeat(4*n_nearest, 1, 1)

    #Get the +- n nearest neighbors
    for i in range(n_nearest):
        expanded[i, :, 0] += (i+1)
        expanded[n_nearest+i, :, 1] += (i+1)
        expanded[2*n_nearest+i, :, 0] -= (i+1)
        expanded[3*n_nearest+i, :, 1] -= (i+1)

    nearest_neighbors_01 = torch.cat([
        check_A_in_B(expanded[i], indices) for i in range(expanded.shape[0])
    ], dim=1)
    return nearest_neighbors_01

import math
class Network(nn.Module):



    def __init__(
            self,
            wires_file='protodunevd-wires-larsoft-v3.json.bz2',
            nfeatures=4,
            time_window=3,
            n_feat_wire = 4,
            detector=0):
        super().__init__()
        with torch.no_grad():
            self.nfeat_post_unet=nfeatures
            self.n_feat_wire = n_feat_wire
            ##Set up the UNets
            self.unets = nn.ModuleList([
                    UNet(n_channels=2, n_classes=nfeatures,
                        batch_norm=True, bilinear=True, padding=True)
                    for i in range(3)
            ])

            self.GNN = GAT(
                2+n_feat_wire, #Input -- testing without passing unets for now
                4, #Hidden channels -- starting small
                2, #N message passes -- starting small
            )

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

            view_base = len(self.coords_face0.views) - 3
            self.ray_crossings_0_01 = self.coords_face0.ray_crossing(view_base + 0, self.good_indices_0_01[:,0], view_base + 1, self.good_indices_0_01[:,1])
            self.ray_crossings_0_12 = self.coords_face0.ray_crossing(view_base + 1, self.good_indices_0_12[:,0], view_base + 2, self.good_indices_0_12[:,1])
            self.ray_crossings_0_20 = self.coords_face0.ray_crossing(view_base + 2, self.good_indices_0_20[:,0], view_base + 0, self.good_indices_0_20[:,1])


            self.all_ray_crossings = torch.cat([
                self.ray_crossings_0_01, self.ray_crossings_0_12, self.ray_crossings_0_20
            ])

            #Neighbors on either face of the anode
            n_nearest = 3
            n_0_01 = len(self.ray_crossings_0_01)
            n_0_12 = len(self.ray_crossings_0_12)
            n_0_20 = len(self.ray_crossings_0_20)
            self.nearest_neighbors_0_01 = get_nn_from_plane_pair(self.good_indices_0_01, n_nearest=n_nearest)
            self.nearest_neighbors_0_12 = get_nn_from_plane_pair(self.good_indices_0_12, n_nearest=n_nearest) + n_0_01
            self.nearest_neighbors_0_20 = get_nn_from_plane_pair(self.good_indices_0_20, n_nearest=n_nearest) + n_0_01 + n_0_12

            #Neighbors between anode faces which are connected by the elec channel
            #TODO


            # self.neighbors = torch.cat([
            #     self.nearest_neighbors_0_01, self.nearest_neighbors_0_12, self.nearest_neighbors_0_20
            # ], dim=1)

            self.neighbors = self.nearest_neighbors_0_01

            #Static edge attributes -- dZ, dY, r=sqrt(dZ**2 + dY**2), dFace
            #TODO  Do things like dWire0, dWire1 make sense for things like cross-pair (i.e. 0,1 and 0,2) neighbors?
            #      Same question for cross face (i.e. 0,1 on face 0 and 0,1 on face 1)
            self.nstatic_edge_attr = 4
            self.static_edges_0_01 = torch.zeros(self.nearest_neighbors_0_01.size(1), self.nstatic_edge_attr)
            self.static_edges_0_01[:, :2] = (
                self.ray_crossings_0_01[self.nearest_neighbors_0_01[0]] -
                self.ray_crossings_0_01[self.nearest_neighbors_0_01[1]]
            ) #dZ, dY
            self.static_edges_0_01[:, 2] = torch.norm(self.static_edges_0_01[:, :2], dim=1) # r
            self.static_edges_0_01[:, 3] = 0 #dFace
            
            #This would be the differeince in electronics channel from plane 0, between the nearest neighbors
            #It's really confusing so maybe think more about implementing
            # self.static_edges_0_01[:, 4] = (
            #     self.face_plane_wires_channels[0,0][self.good_indices_0_01[self.nearest_neighbors_0_01[0]][0]] -
            #     self.face_plane_wires_channels[0,0][self.good_indices_0_01[self.nearest_neighbors_0_01[1]][0]]
            # ) 
            
            # self.static_edges_0_01[:, 4] = 1 #dPlane
            # self.static_edges_0_01[:, 5] = self.good_indices_0_01
            # self.nearest_neighbors_0_01[0] - self.nearest_neighbors_0_01[1] #dWire0
            # self.static_edges_0_01[:, 6] = (
            # implement dWire
            # )

            self.static_edges = self.static_edges_0_01

            self.nchans = [476, 476, 292, 292]

    def forward(self, x):
        '''
        Input data is assumed to be of shape (nbatch, nfeatures, nchannels, nticks)
        '''
        input_shape = x.shape
        nbatches = x.shape[0]
        nticks = x.shape[-1]

        the_device = x.device
        print('Pre unet', x.shape)
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

        print('Post unet', x.shape)

        n_feat_base = x.shape[1]
        
        #For ease
        x = x.permute(0,1,3,2)

        #Convert from channels to wires (values duped for common elec chan)
        #Also expand features to include 'meta' features i.e. wire seg number, elec channel
        n_feat_wire = 4
        new_shape = (x.shape[0], n_feat_base+n_feat_wire, x.shape[2], )
        as_wires_f0_p0 = torch.zeros(new_shape + (len(self.face_plane_wires_channels[0,0]),))
        as_wires_f0_p0[:, :n_feat_base, :, self.face_plane_wires_channels[0,0][:,0]] = x[..., self.face_plane_wires_channels[0,0][:,1]]

        as_wires_f0_p1 = torch.zeros(new_shape + (len(self.face_plane_wires_channels[0,1]),))
        as_wires_f0_p1[:, :n_feat_base, :, self.face_plane_wires_channels[0,1][:,0]] = x[..., self.face_plane_wires_channels[0,1][:,1]]
        
        as_wires_f0_p2 = torch.zeros(new_shape + (len(self.face_plane_wires_channels[0,2]),))
        as_wires_f0_p2[:, :n_feat_base, :, self.face_plane_wires_channels[0,2][:,0]] = x[..., self.face_plane_wires_channels[0,2][:,1]]

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

        #Anode face
        as_wires_f0_p0[..., n_feat_base + 2] = 0
        as_wires_f0_p1[..., n_feat_base + 2] = 0
        as_wires_f0_p2[..., n_feat_base + 2] = 0

        #Wire plane
        as_wires_f0_p0[..., n_feat_base + 3] = 0
        as_wires_f0_p1[..., n_feat_base + 3] = 1
        as_wires_f0_p2[..., n_feat_base + 3] = 2

        #Could add more things: i.e. channel RMS over readout window.
        #Worth some thought and tests
        #Maybe the number of electrically-connected wire segments on either side
        
        #Now set up our 2-channel crossings -- these will be our GNN nodes
        print('Crossers 01:', self.good_indices_0_01.shape)
        crossings_01 = torch.cat([
            as_wires_f0_p0[:, :, self.good_indices_0_01[:,0], :],
            as_wires_f0_p1[:, :, self.good_indices_0_01[:,1], :],
            self.ray_crossings_0_01.view(1, 1, -1, 2).repeat(nbatches, nticks, 1, 1), #locations of crossings
            torch.arange(nticks).view(1, -1, 1, 1).repeat(nbatches, 1, self.good_indices_0_01.size(0), 1), #tick number
        ], dim=-1)

        print(crossings_01.shape)
        nfeat = crossings_01.shape[-1]
        
        # torch.save(crossings_01[:,:,:,(2,6)], 'crossings_01.pt')
        # torch.save(self.good_indices_0_01, 'good_indices_01.pt')

        crossings_12 = torch.cat([
           as_wires_f0_p1[:, :, self.good_indices_0_12[:,0], :],
           as_wires_f0_p2[:, :, self.good_indices_0_12[:,1], :],
           self.ray_crossings_0_12.view(1, 1, -1, 2).repeat(nbatches, nticks, 1, 1), #locations of crossings
           torch.arange(nticks).view(1, -1, 1, 1).repeat(nbatches, 1, self.good_indices_0_12.size(0), 1), #tick number
        ], dim=-1)

        crossings_20 = torch.cat([
           as_wires_f0_p2[:, :, self.good_indices_0_20[:,0], :],
           as_wires_f0_p0[:, :, self.good_indices_0_20[:,1], :],
           self.ray_crossings_0_20.view(1, 1, -1, 2).repeat(nbatches, nticks, 1, 1), #locations of crossings
           torch.arange(nticks).view(1, -1, 1, 1).repeat(nbatches, 1, self.good_indices_0_20.size(0), 1), #tick number
        ], dim=-1)

        print('X 01:', crossings_01.shape)
        ncross_01 = crossings_01.shape[-2]
        ncross_12 = crossings_12.shape[-2]
        ncross_20 = crossings_20.shape[-2]
        
        
        all_crossings = crossings_01
        # ncross = ncross_01 + ncross_12 + ncross_20
        ncross = ncross_01

        #WHEN BUILDING UP THE TIME WINDOW FUNCTIONALITY
        # TRY TO MAKE IT SO THAT YOU CAN JUST SET TIME WINDOW = 1
        # THIS WILL BE USEFUL AS A HYPER PARAMETER & FOR ABLATION STUDIES

        for i in range(all_crossings.size(1)):
            hi = i+1 if i == all_crossings.size(1)-1 else i + int((self.time_window-1)/2) + 1
            low = 0 if i == 0 else i - int((self.time_window-1)/2)

            window = all_crossings[:, low:hi, ...].view(nbatches, -1, nfeat)
            # print(i, low, hi, window.shape)
            #Window neighbors includes the nearest neighbors within the time tick
            # as well as the common crossing points between the time ticks

            #in-tick crossings
            window_neighbors = self.neighbors.repeat(1, hi-low)
            
            #between ticks
            # tick_neighbors = []
            tick_neighbors = torch.arange(ncross).unsqueeze(0).repeat(2,(hi-low)**2)

            for i in range(hi-low):
                window_neighbors[:, i*ncross:(i+1)*ncross] += (i*ncross)
                for j in range(hi-low):
                    # if i == j: continue #Self-message-pass
                    # tick_neighbors.append(torch.cat([
                    #     torch.arange(i*ncross, (i+1)*ncross).unsqueeze(0), torch.arange(ncross*j, ncross*(j+1)).unsqueeze(0)
                    # ], dim=0))
                    tick_neighbors[0, ncross*(j*(hi-low) + i):ncross*(j*(hi-low) + (i+1))] += i*ncross
                    tick_neighbors[1, ncross*(j*(hi-low) + i):ncross*(j*(hi-low) + (i+1))] += j*ncross

            # print(tick_neighbors)
            #TODO -- get unique between tick neighbors
            # tick_neighbors = torch.cat(tick_neighbors, dim=1)
            # print(tick_neighbors)
            all_neighbors = torch.cat([window_neighbors, tick_neighbors], dim=1)
            
            # all_neighbors = window_neighbors
            
            n_edge_attr = self.nstatic_edge_attr + 1 #+1 for tick
            edge_attr = torch.zeros(all_neighbors.size(1), n_edge_attr)
            window = window.reshape(-1, nfeat)
            
            #TODO -- consider batching
            edge_attr[:window_neighbors.size(1), :-1] = self.static_edges.view(self.neighbors.size(1), -1).repeat(1*(hi-low), 1)
            base = window_neighbors.size(1)
            for i in range(hi-low):
                for j in range(hi-low):
                    ind_0 = (base + ncross*(j*(hi-low) + i))
                    ind_1 = ind_0 + ncross
                    edge_attr[ind_0:ind_1, -1] = (i-j)
            
            
            # print(edge_attr[0])




        return 1

