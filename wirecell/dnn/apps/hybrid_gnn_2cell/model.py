import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch_geometric.data import Data
from torch_geometric.nn import GAT
from wirecell.dnn.models.unet import UNet
from wirecell.raygrid.coordinates import Coordinates
from wirecell.raygrid import crossover as xover
from wirecell.util.wires import schema, persist
import torch.utils.checkpoint as checkpoint

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

def fill_window(w, nfeat, first_wires, second_wires, indices, crossings, coords):
    w[..., :(nfeat)] = first_wires[:, :, indices[:,0], :]
    w[..., (nfeat):2*(nfeat)] = second_wires[:, :, indices[:,1], :]
    start = 2*(nfeat)
    end = start+2
    w[..., start:end] = crossings.view(1, 1, -1, 2).repeat(w.shape[0], w.shape[1], 1, 1).to(w.device)
    w[..., start:end] /= torch.norm(coords.bounding_box, dim=1).to(w.device)
def get_nn_from_plane_pair(indices, n_nearest=3):

    #We are creating +- n_nearest neighbors --> 2 for each plane in the pair -> 4
    expanded = indices.unsqueeze(0).repeat(4*n_nearest, 1, 1)

    #Get the +- n nearest neighbors along each wire
    for i in range(n_nearest):
        expanded[i, :, 0] += (i+1)
        expanded[n_nearest+i, :, 1] += (i+1)
        expanded[2*n_nearest+i, :, 0] -= (i+1)
        expanded[3*n_nearest+i, :, 1] -= (i+1)

    # Now, we ned to get this in terms of the crossing indices.
    # check_A_in_B looks for the pairs we created in "expanded"
    # and outputs (index within A, index within B).
    #
    # So it ties the pair in question (index in A) to the +-nth crossing (index in B)
    #
    # Outside of this, we account for the fact that we concatenate all plane 
    # pairs into one large crossing tensor
    nearest_neighbors_01 = torch.cat([
        check_A_in_B(expanded[i], indices) for i in range(expanded.shape[0])
    ], dim=1)
    return nearest_neighbors_01

def get_nn_third_plane(
        coords, 
        indices_ij,
        indices_jk,
        indices_ki,
        i, j, k, n_nearest=3):
    '''
    For this, we take the wire crossings between planes i & j and find the pitch index
    in k which matches this crossing point.

    We then say (k',i) and (j,k') are neighbors, where k' is in [k, k +- N]

    The index we find in k is not determined to be within the 'valid indices' of either (j,k) or (k,i)
    (these are provided), so we have to use check_A_in_B. 

    This checks whether the pair we are considering i.e. (j,k') is within indices_jk. It then returns the
    index within the indices_ij and the index within indices_jk.

    In other words, the edge is formed by saying pair (i, j) is connected to (j, k') via these 2 indices.
    '''
    
    #TODO -- consider clamping
    base = len(coords.views) - 3
    #Crossing point indices in the third plane
    in_k = xover.get_indices(coords, indices_ij, i+base, j+base, k+base)

    def combine(a, b):
        return torch.cat(
            [a.unsqueeze(0).view(-1,1), b.unsqueeze(0).view(-1,1)],
            dim=1
        )

    ki_pairs = [check_A_in_B(combine(in_k, indices_ij[:, 0]), indices_ki)]
    jk_pairs = [check_A_in_B(combine(indices_ij[:, 1], in_k), indices_jk)]
    
    for inear in range(n_nearest):
        for sign in [+1, -1]:
            ki_pairs.append(
                check_A_in_B(
                    combine(in_k+sign*(inear+1), indices_ij[:, 0]),
                    indices_ki
                )
            )
            jk_pairs.append(
                check_A_in_B(
                    combine(indices_ij[:, 1], in_k+sign*(inear+1)),
                    indices_jk
                )
            )

        print('3rd plane:', ki_pairs[-1].shape, jk_pairs[-1].shape, ki_pairs[-1][:5], jk_pairs[-1][:5])
    return {(k,i):torch.cat(ki_pairs, dim=1), (j,k):torch.cat(jk_pairs, dim=1)}
    # expanded = indices.unsqueeze(0).repeat(4*n_nearest, 1, 1)
    # #Get the +- n nearest neighbors
    # for i in range(n_nearest):
    #     expanded[i, :, 0] += (i+1)
    #     expanded[n_nearest+i, :, 1] += (i+1)
    #     expanded[2*n_nearest+i, :, 0] -= (i+1)
    #     expanded[3*n_nearest+i, :, 1] -= (i+1)

    # nearest_neighbors_01 = torch.cat([
    #     check_A_in_B(expanded[i], indices) for i in range(expanded.shape[0])
    # ], dim=1)
    # return nearest_neighbors_01

import math
class Network(nn.Module):
    # @torch.autocast('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16) #TODO -- Fix this
    def __init__(
            self,
            wires_file='protodunevd-wires-larsoft-v3.json.bz2',
            n_unet_features=16,
            time_window=1,
            n_feat_wire =0,
            detector=0,
            n_input_features=1,
            skip_unets=True,
            skip_GNN=False,
            out_channels=4):
        super().__init__()
        self.nfeat_post_unet=n_unet_features
        self.n_feat_wire = n_feat_wire
        self.n_input_features=n_input_features
        self.skip_unets=skip_unets
        ##Set up the UNets
        self.unets = nn.ModuleList([
                UNet(n_channels=n_input_features, n_classes=n_unet_features,
                    batch_norm=True, bilinear=True, padding=True)
                for i in range(3)
        ])
        self.skip_GNN=skip_GNN
        if skip_unets:
            n_unet_features=n_input_features

        self.GNN = GAT(
            2*(n_unet_features + n_feat_wire) + 2, #Input
            1, #Hidden channels -- starting small
            1, #N message passes -- starting small
            out_channels=out_channels,
        )
        self.out_channels=out_channels
        self.mlp = nn.Linear((n_unet_features if skip_GNN else out_channels), 1)
        with torch.no_grad():

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
            #This comes from channels_X in one of the input files
            #The index of the array is the 'global' channel number -- index into frame array.
            #The value of the array is the wire.channel -- larsoft channel ID 
            chanmap_npy = np.load('chanmap_1536.npy')

            #maps from chanident to index in input arrays
            chanmap = {c:i for i, c in chanmap_npy}
            # for i, c in chanmap_npy: print(i,c)


            #Build the map to go between wire segments & channels 
            self.face_plane_wires_channels = {}
            for i, face in enumerate(self.faces):
                for jj, j in enumerate(face.planes):
                    plane = store.planes[j]
                    wire_chans = []
                    for wi in plane.wires:
                        wire = store.wires[wi]
                        wire_chans.append([wire.ident, chanmap[wire.channel]]) #convert from larsoft
                    self.face_plane_wires_channels[(i,jj)] = torch.tensor(wire_chans, dtype=int)
                    print('Made fpwc:', i, jj, self.face_plane_wires_channels[(i,jj)].shape)
            

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

            self.good_indices_1_01 = xover.get_good_crossers(self.coords_face1, 0, 1, self.nwires_1)
            self.good_indices_1_12 = xover.get_good_crossers(self.coords_face1, 1, 2, self.nwires_1)
            self.good_indices_1_20 = xover.get_good_crossers(self.coords_face1, 2, 0, self.nwires_1)

            view_base = len(self.coords_face0.views) - 3
            self.ray_crossings_0_01 = self.coords_face0.ray_crossing(view_base + 0, self.good_indices_0_01[:,0], view_base + 1, self.good_indices_0_01[:,1])
            self.ray_crossings_0_12 = self.coords_face0.ray_crossing(view_base + 1, self.good_indices_0_12[:,0], view_base + 2, self.good_indices_0_12[:,1])
            self.ray_crossings_0_20 = self.coords_face0.ray_crossing(view_base + 2, self.good_indices_0_20[:,0], view_base + 0, self.good_indices_0_20[:,1])
            self.ray_crossings_1_01 = self.coords_face1.ray_crossing(view_base + 0, self.good_indices_1_01[:,0], view_base + 1, self.good_indices_1_01[:,1])
            self.ray_crossings_1_12 = self.coords_face1.ray_crossing(view_base + 1, self.good_indices_1_12[:,0], view_base + 2, self.good_indices_1_12[:,1])
            self.ray_crossings_1_20 = self.coords_face1.ray_crossing(view_base + 2, self.good_indices_1_20[:,0], view_base + 0, self.good_indices_1_20[:,1])


            #Need this?
            # self.all_ray_crossings = torch.cat([
            #     self.ray_crossings_0_01, self.ray_crossings_0_12, self.ray_crossings_0_20
            # ])

            #Neighbors on either face of the anode
            n_nearest = 1
            n_0_01 = len(self.ray_crossings_0_01)
            n_0_12 = len(self.ray_crossings_0_12)
            n_0_20 = len(self.ray_crossings_0_20)
            n_0_total = n_0_01 + n_0_12 + n_0_20

            n_1_01 = len(self.ray_crossings_1_01)
            n_1_12 = len(self.ray_crossings_1_12)
            n_1_20 = len(self.ray_crossings_1_20)

            nearest_neighbors_0_01 = get_nn_from_plane_pair(self.good_indices_0_01, n_nearest=n_nearest)
            nearest_neighbors_0_12 = get_nn_from_plane_pair(self.good_indices_0_12, n_nearest=n_nearest)
            nearest_neighbors_0_20 = get_nn_from_plane_pair(self.good_indices_0_20, n_nearest=n_nearest)
            nearest_neighbors_1_01 = get_nn_from_plane_pair(self.good_indices_1_01, n_nearest=n_nearest)
            nearest_neighbors_1_12 = get_nn_from_plane_pair(self.good_indices_1_12, n_nearest=n_nearest)
            nearest_neighbors_1_20 = get_nn_from_plane_pair(self.good_indices_1_20, n_nearest=n_nearest)

            self.nstatic_edge_attr = 4
            n_nearest_third_plane = 0
            third_plane_neighbors_0_012 = get_nn_third_plane(
                self.coords_face0,
                self.good_indices_0_01, self.good_indices_0_12, self.good_indices_0_20,
                0, 1, 2, n_nearest=n_nearest_third_plane)
            third_plane_neighbors_0_120 = get_nn_third_plane(
                self.coords_face0,
                self.good_indices_0_12, self.good_indices_0_20, self.good_indices_0_01,
                1, 2, 0, n_nearest=n_nearest_third_plane)
            third_plane_neighbors_0_201 = get_nn_third_plane(
                self.coords_face0,
                self.good_indices_0_20, self.good_indices_0_01, self.good_indices_0_12,
                2, 0, 1, n_nearest=n_nearest_third_plane)
            

            plane3_edges_01_12 = self.make_edge_attr(
                third_plane_neighbors_0_012[(1,2)],
                self.nstatic_edge_attr,
                self.ray_crossings_0_01, self.ray_crossings_0_12
            )
            plane3_edges_01_20 = self.make_edge_attr(
                third_plane_neighbors_0_012[(2,0)],
                self.nstatic_edge_attr,
                self.ray_crossings_0_01, self.ray_crossings_0_20
            )
            
            plane3_edges_12_20 = self.make_edge_attr(
                third_plane_neighbors_0_120[(2,0)],
                self.nstatic_edge_attr,
                self.ray_crossings_0_12, self.ray_crossings_0_20
            )
            plane3_edges_12_01 = self.make_edge_attr(
                third_plane_neighbors_0_120[(0,1)],
                self.nstatic_edge_attr,
                self.ray_crossings_0_12, self.ray_crossings_0_01
            )
            
            plane3_edges_20_01 = self.make_edge_attr(
                third_plane_neighbors_0_201[(0,1)],
                self.nstatic_edge_attr,
                self.ray_crossings_0_20, self.ray_crossings_0_01
            )
            plane3_edges_20_12 = self.make_edge_attr(
                third_plane_neighbors_0_201[(1,2)],
                self.nstatic_edge_attr,
                self.ray_crossings_0_20, self.ray_crossings_0_12
            )

            #Account for the 'global' crossing indices
            third_plane_neighbors_0_012[(1,2)][0,:] += 0 #For clarity/explicitness
            third_plane_neighbors_0_012[(2,0)][0,:] += 0 #For clarity/explicitness
            third_plane_neighbors_0_012[(1,2)][1,:] += n_0_01
            third_plane_neighbors_0_012[(2,0)][1,:] += n_0_01 + n_0_12
            
            third_plane_neighbors_0_120[(2,0)][0,:] += n_0_01
            third_plane_neighbors_0_120[(0,1)][0,:] += n_0_01
            third_plane_neighbors_0_120[(2,0)][1,:] += n_0_01 + n_0_12
            third_plane_neighbors_0_120[(0,1)][1,:] += 0 #For clarity/explicitness
            
            third_plane_neighbors_0_201[(0,1)][0,:] += n_0_01 + n_0_12
            third_plane_neighbors_0_201[(1,2)][0,:] += n_0_01 + n_0_12
            third_plane_neighbors_0_201[(0,1)][1,:] += 0 #For clarity/explicitness
            third_plane_neighbors_0_201[(1,2)][1,:] += n_0_01
            print('Made 3rd plane neighbors')
            # time.sleep(10)

            #Neighbors between anode faces which are connected by the elec channel?
            #TODO


            self.neighbors = torch.cat([
                nearest_neighbors_0_01,
                (nearest_neighbors_0_12 + n_0_01),
                (nearest_neighbors_0_20 + n_0_01 + n_0_12),
                
                nearest_neighbors_1_01,
                (nearest_neighbors_1_12 + n_0_total + n_1_01),
                (nearest_neighbors_1_20 + n_0_total + n_1_01 + n_1_12),

                third_plane_neighbors_0_012[(1,2)],
                third_plane_neighbors_0_012[(2,0)],
                third_plane_neighbors_0_120[(2,0)],
                third_plane_neighbors_0_120[(0,1)],
                third_plane_neighbors_0_201[(0,1)],
                third_plane_neighbors_0_201[(1,2)],
            ], dim=1)

            #Static edge attributes -- dZ, dY, r=sqrt(dZ**2 + dY**2), dFace
            #TODO  Do things like dWire0, dWire1 make sense for things like cross-pair (i.e. 0,1 and 0,2) neighbors?
            #      Same question for cross face (i.e. 0,1 on face 0 and 0,1 on face 1)
            


            static_edges_0_01 = self.make_edge_attr(
                nearest_neighbors_0_01, self.nstatic_edge_attr,
                self.ray_crossings_0_01, self.ray_crossings_0_01,
                0)

            static_edges_0_12 = self.make_edge_attr(
                nearest_neighbors_0_12, self.nstatic_edge_attr,
                self.ray_crossings_0_12, self.ray_crossings_0_12,
                0)

            static_edges_0_20 = self.make_edge_attr(
                nearest_neighbors_0_20, self.nstatic_edge_attr,
                self.ray_crossings_0_20, self.ray_crossings_0_20,
                0)
            
            static_edges_1_01 = self.make_edge_attr(
                nearest_neighbors_1_01, self.nstatic_edge_attr,
                self.ray_crossings_1_01, self.ray_crossings_1_01,
                0)
            static_edges_1_12 = self.make_edge_attr(
                nearest_neighbors_1_12, self.nstatic_edge_attr,
                self.ray_crossings_1_12, self.ray_crossings_1_12,
                0)
            static_edges_1_20 = self.make_edge_attr(
                nearest_neighbors_1_20, self.nstatic_edge_attr,
                self.ray_crossings_1_20, self.ray_crossings_1_20,
                0)            
            

            self.static_edges = torch.cat([
                static_edges_0_01,
                static_edges_0_12,
                static_edges_0_20,
                static_edges_1_01,
                static_edges_1_12,
                static_edges_1_20,
                plane3_edges_01_12,
                plane3_edges_01_20,
                plane3_edges_12_20,
                plane3_edges_12_01,
                plane3_edges_20_01,
                plane3_edges_20_12,
            ])

            self.static_edges[..., :2] /= torch.norm(self.coords_face0.bounding_box, dim=1)
            self.static_edges[:, 2] = torch.norm(self.static_edges[:, :2], dim=1)

            print('Len static edges & neighbors:',  self.static_edges.shape, self.neighbors.shape)

            self.nchans = [476, 476, 292, 292]

            # self.sigmoid = nn.Sigmoid()
            self.save = True

    def make_edge_attr(self, neighbors, nattr, crossings_0, crossings_1, dface=0):
        edge_attrs = torch.zeros(neighbors.size(1), nattr)
        edge_attrs[:, :2] = (
            crossings_0[neighbors[0]] -
            crossings_1[neighbors[1]]
        ) #dZ, dY
        edge_attrs[:, 2] = torch.norm(edge_attrs[:, :2], dim=1) # r
        edge_attrs[:, 3] = dface #dFace
        return edge_attrs
    
    def scatter_to_chans(self, y, nbatches, nchannels, the_device):
        #TODO -- check size of y etc
        temp_out = torch.zeros(nbatches, nchannels, y[0].size(-1)).to(the_device)

        to_scatter = [
            #plane0
            [y[0], self.face_plane_wires_channels[(0,0)][self.good_indices_0_01[:,0]][:,1]],
            [y[2], self.face_plane_wires_channels[(0,0)][self.good_indices_0_20[:,1]][:,1]],
            
            #plane1
            [y[0], self.face_plane_wires_channels[(0,1)][self.good_indices_0_01[:,1]][:,1]],
            [y[1], self.face_plane_wires_channels[(0,1)][self.good_indices_0_12[:,0]][:,1]],
            
            #plane2
            [y[1], self.face_plane_wires_channels[(0,2)][self.good_indices_0_12[:,1]][:,1]],
            [y[2], self.face_plane_wires_channels[(0,2)][self.good_indices_0_20[:,0]][:,1]],

            #plane0
            [y[3], self.face_plane_wires_channels[(1,0)][self.good_indices_1_01[:,0]][:,1]],
            [y[5], self.face_plane_wires_channels[(1,0)][self.good_indices_1_20[:,1]][:,1]],
            
            #plane1
            [y[3], self.face_plane_wires_channels[(1,1)][self.good_indices_1_01[:,1]][:,1]],
            [y[4], self.face_plane_wires_channels[(1,1)][self.good_indices_1_12[:,0]][:,1]],
            
            #plane2
            [y[4], self.face_plane_wires_channels[(1,2)][self.good_indices_1_12[:,1]][:,1]],
            [y[5], self.face_plane_wires_channels[(1,2)][self.good_indices_1_20[:,0]][:,1]],
        ]

        for yi, indices in to_scatter:
            # torch_scatter.scatter_add(
            # torch_scatter.scatter_max(
            torch_scatter.scatter_mean(
                yi,
                indices,
                out=temp_out,
                dim=1
            )

        return temp_out
    def make_wires(self, x, low, hi, nfeat, face, plane):
        wire_chans = self.face_plane_wires_channels[(face, plane)]
        wires = torch.zeros((x.shape[0], (hi-low), len(wire_chans), nfeat)).to(x.device)
        wires[:, :, wire_chans[:, 0], :x.shape[-1]] = x[:, low:hi, wire_chans[:,1], :]
        # wires[..., x.shape[-1]] = plane
        # wires[..., x.shape[-1]+1] = face
        return wires
    
    def forward(self, x):
        '''
        Input data is assumed to be of shape (nbatch, nfeatures, nchannels, nticks)
        '''
        input_shape = x.shape
        nbatches = x.shape[0]
        nticks = x.shape[-1]
        nchannels = x.shape[-2]

        the_device = x.device
        print('Pre unet', x.shape)

        if not self.skip_unets:
            if self.save:
                torch.save(x, 'input_test.pt')
            xs = [
                x[:, :, (0 if i == 0 else sum(self.nchans[:i])):sum(self.nchans[:i+1]), :]
                for i, nc in enumerate(self.nchans)
            ]
            for x in xs: print(x.shape)

            #Pass through the unets
            xs = [
                # self.unets[(i if i < 3 else 2)](xs[i]) for i in range(len(xs))
                checkpoint.checkpoint(
                    self.unets[(i if i < 3 else 2)],
                    xs[i]
                ) for i in range(len(xs))
            ]

            print('passed through unets')
            for x in xs: print(x.shape)

            #Cat to get into global channel number shape
            x = torch.cat(xs, dim=2)
            if self.save:
                torch.save(x, 'post_unet_test.pt')
            print('Post unet', x.shape)


        

        if self.skip_GNN:
            x = x.permute(0,3,2,1)
            return self.mlp(x).permute(0,3,2,1)
        else:
            n_feat_base = x.shape[1]
            nticks_orig = x.size(-1)
            #For ease
            #batch, tick, channels, features
            to_pad = int((self.time_window-1)/2)
            x = F.pad(x, (to_pad, to_pad))
            nticks = x.size(-1)
            print(x.shape)
            x = x.permute(0,3,2,1)
            #Convert from channels to wires (values duped for common elec chan)
            #Also expand features to include 'meta' features i.e. wire seg number, elec channel
            # n_feat_wire = 4
            # nchan_f0_p0 = len(self.face_plane_wires_channels[0,0])
            # nchan_f0_p1 = len(self.face_plane_wires_channels[0,1])
            # nchan_f0_p2 = len(self.face_plane_wires_channels[0,2])

            these_nfeats = n_feat_base + self.n_feat_wire

            for ij, tensor in self.face_plane_wires_channels.items():
                self.face_plane_wires_channels[ij] = tensor.to(the_device)
            
            nfeat = 2*(these_nfeats) + 2#
            
            ncross_01 = self.good_indices_0_01.size(0)
            ncross_12 = self.good_indices_0_12.size(0)
            ncross_20 = self.good_indices_0_20.size(0)
            ncross_1_01 = self.good_indices_1_01.size(0)
            ncross_1_12 = self.good_indices_1_12.size(0)
            ncross_1_20 = self.good_indices_1_20.size(0)
            ncrosses = [ncross_01, ncross_12, ncross_20, ncross_1_01, ncross_1_12, ncross_1_20]
            # ncrosses = [ncross_01, ncross_12, ncross_20]
            ranges = [[sum(ncrosses[:i]), sum(ncrosses[:i+1])] for i in range(len(ncrosses))]
            ncross = ncross_01 + ncross_12 + ncross_20 + ncross_1_01 + ncross_1_12 + ncross_1_20
            print(ncross_01, ncross_12, ncross_20, ncross)

            #in-tick crossings
            dt = self.time_window
            n_window_neighbors = self.neighbors.size(1)
            new_window_neighbors_size = n_window_neighbors*dt
            all_neighbors = torch.zeros(2, new_window_neighbors_size + ncross*((dt)**2), dtype=int).to(the_device)
            all_neighbors[:, :n_window_neighbors*(dt)] = self.neighbors.repeat(1, dt).to(the_device)
            all_neighbors[:, new_window_neighbors_size:] = torch.arange(ncross).unsqueeze(0).repeat(2,(dt)**2).to(the_device)

            for i in range(dt):
                all_neighbors[:, i*n_window_neighbors:(i+1)*n_window_neighbors] += (i*ncross)
                # print(i, ncross*i, n_window_neighbors, i*n_window_neighbors, (i+1)*n_window_neighbors)

                for j in range(dt):
                    all_neighbors[0, new_window_neighbors_size + (ncross*(j*(dt) + i)):new_window_neighbors_size + (ncross*(j*(dt) + (i+1)))] += i*ncross
                    all_neighbors[1, new_window_neighbors_size + (ncross*(j*(dt) + i)):new_window_neighbors_size + (ncross*(j*(dt) + (i+1)))] += j*ncross

            n_edge_attr = self.nstatic_edge_attr + 1 #+1 for tick
            edge_attr = torch.zeros(all_neighbors.size(1), n_edge_attr).to(the_device)
            #TODO -- consider batching
            edge_attr[:new_window_neighbors_size, :-1] = self.static_edges.view(self.neighbors.size(1), -1).repeat(1*(dt), 1).to(the_device)
            base = new_window_neighbors_size

            print('Edges & neighbors:', all_neighbors.size(), edge_attr.size())

            for i in range(dt):
                for j in range(dt):
                    ind_0 = (base + ncross*(j*(dt) + i))
                    ind_1 = ind_0 + ncross
                    edge_attr[ind_0:ind_1, -1] = (i-j)
                    
            out = torch.zeros(x.shape[0], 1, nticks_orig, nchannels).to(x.device)
            roundabout = torch.zeros(x.shape[0], nfeat, nticks_orig, nchannels).to(x.device)
            for tick in range(nticks_orig):
                low = tick
                hi = low + 2*to_pad+1

                #NEW AS WIRES

                    
                as_wires_f0_p0 = self.make_wires(x, low, hi, these_nfeats, 0, 0)
                as_wires_f0_p1 = self.make_wires(x, low, hi, these_nfeats, 0, 1)
                as_wires_f0_p2 = self.make_wires(x, low, hi, these_nfeats, 0, 2)
                as_wires_f1_p0 = self.make_wires(x, low, hi, these_nfeats, 1, 0)
                as_wires_f1_p1 = self.make_wires(x, low, hi, these_nfeats, 1, 1)
                as_wires_f1_p2 = self.make_wires(x, low, hi, these_nfeats, 1, 2)
                ######################


                window = torch.zeros(
                    nbatches,
                    dt,
                    ncross,
                    nfeat,
                ).to(the_device)


                cross_start = 0
                # cross_end = ncross_01
                cross_end = 0
                window_infos = [
                    [as_wires_f0_p0, as_wires_f0_p1, self.good_indices_0_01, self.ray_crossings_0_01, self.coords_face0],
                    [as_wires_f0_p1, as_wires_f0_p2, self.good_indices_0_12, self.ray_crossings_0_12, self.coords_face0],
                    [as_wires_f0_p2, as_wires_f0_p0, self.good_indices_0_20, self.ray_crossings_0_20, self.coords_face0],
                    [as_wires_f1_p0, as_wires_f1_p1, self.good_indices_1_01, self.ray_crossings_1_01, self.coords_face1],
                    [as_wires_f1_p1, as_wires_f1_p2, self.good_indices_1_12, self.ray_crossings_1_12, self.coords_face1],
                    [as_wires_f1_p2, as_wires_f1_p0, self.good_indices_1_20, self.ray_crossings_1_20, self.coords_face1],
                ]
                for info in window_infos:
                    cross_end += len(info[2])
                    fill_window(
                        window[..., cross_start:cross_end, :],
                        these_nfeats,
                        *info
                    )
                    cross_start = cross_end

                window = window.reshape(-1, nfeat)

                ##Saving roundabout
                roundabout_y = window.reshape(nbatches, self.time_window, -1, nfeat)[:, int((self.time_window-1)/2), ...]
                roundabout_y = [
                    roundabout_y[:, r0:r1, :] for r0, r1 in ranges
                ]
                # roundabout_y = [
                #     roundabout_y[:, :ncross_01, :],
                #     roundabout_y[:, ncross_01:ncross_01+ncross_12, :],
                #     roundabout_y[:, ncross_01+ncross_12:ncross_01+ncross_12+ncross_20:, :],
                # ]
                roundabout_y = self.scatter_to_chans(roundabout_y, nbatches, nchannels, the_device)
                roundabout[:, :, tick, :] = roundabout_y.permute(0, 2, 1)
                ################


                # y = self.GNN(window, all_neighbors, edge_attr=edge_attr)
                # self.GNN = self.GNN.to('cuda:1')
                # y = self.GNN(window.to('cuda:1'), all_neighbors.to('cuda:1'), edge_attr=edge_attr.to('cuda:1')).to('cuda:0')
                y = checkpoint.checkpoint(
                    self.GNN,
                    window,
                    all_neighbors,
                    edge_attr,
                )

                

                #Just get out the middle element
                y = y.reshape(nbatches, self.time_window, -1, self.out_channels)[:, int((self.time_window-1)/2), ...]

                # y = [
                #     y[:, :ncross_01, :],
                #     y[:, ncross_01:ncross_01+ncross_12, :],
                #     y[:, -ncross_20:, :],
                # ]
                y = [
                    y[:, r0:r1, :] for r0, r1 in ranges
                ]

                # temp_out = torch.zeros(nbatches, nchannels, self.out_channels).to(the_device)
                temp_out = self.scatter_to_chans(y, nbatches, nchannels, the_device)
                

                #batch, feat, channel
                # out = self.sigmoid(self.mlp(out)).view(1, 1, -1)
                # print(temp_out.size())
                # print(out.size())
                temp_out = self.mlp(temp_out).view(1,1,-1)
                out[:, 0, tick, :] = temp_out
        

        if self.save:
            torch.save(roundabout, 'roundabout_test.pt')
            torch.save(out, 'out_test.pt')
            print('Saved')
            self.save = False
        return out.permute(0, 1, 3, 2)

