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

# def check_A_in_B(A, B):
#     B_map = {
#         tuple(row.tolist()): i
#         for i, row in enumerate(B)
#     }
#     A_indices_in_B = []
#     A_indices = []
#     for i, row_a in enumerate(A):
#         row_tuple = tuple(row_a.tolist())
        
#         if row_tuple in B_map:
#             # Found the row: append the index from B
#             A_indices_in_B.append(B_map[row_tuple])
#             A_indices.append(i)
#     A_indices_in_B_tensor = torch.tensor(A_indices_in_B, dtype=torch.long)
#     A_indices_tensor = torch.tensor(A_indices, dtype=torch.long)
#     return torch.cat([
#         A_indices_tensor.unsqueeze(0),
#         A_indices_in_B_tensor.unsqueeze(0),
#     ])



# def get_nn_from_plane_triplet(indices, nwires, n_nearest=3, granularity=1):

#     #We are creating +- n_nearest neighbors --> 2 for each plane in the pair -> 6
#     expanded = indices.unsqueeze(0).repeat(6*n_nearest, 1, 1)

#     #Get the +- n nearest neighbors along each wire
#     for i in range(n_nearest):
#         expanded[i, :, 0] += (i+1)*granularity
#         expanded[n_nearest+i, :, 1] += (i+1)*granularity
#         expanded[2*n_nearest+i, :, 2] += (i+1)*granularity
        
#         expanded[3*n_nearest+i, :, 0] -= (i+1)*granularity
#         expanded[4*n_nearest+i, :, 1] -= (i+1)*granularity
#         expanded[5*n_nearest+i, :, 2] -= (i+1)*granularity

#     expanded = torch.clamp(expanded, min=torch.tensor([0,0,0], dtype=int), max=torch.tensor(nwires, dtype=int))

#     # Now, we ned to get this in terms of the crossing indices.
#     # check_A_in_B looks for the pairs we created in "expanded"
#     # and outputs (index within A, index within B).
#     #
#     # So it ties the pair in question (index in A) to the +-nth crossing (index in B)
#     #
#     # Outside of this, we account for the fact that we concatenate all plane 
#     # pairs into one large crossing tensor
#     nearest_neighbors_01 = torch.cat([
#         check_A_in_B(expanded[i], indices) for i in range(expanded.shape[0])
#     ], dim=1)
#     return nearest_neighbors_01


# def get_nn_from_plane_triplet_fixed(indices, nwires, negative=False, n_nearest=3, granularity=1):

#     #We are creating +- n_nearest neighbors --> 2 for each plane in the pair -> 6
#     expanded = indices.unsqueeze(0).repeat(n_nearest**3, 1, 1)

#     #Get the +- n nearest neighbors along each wire
#     for i in range(n_nearest):
#         for j in range(n_nearest):
#             for k in range(n_nearest):
#                 expanded[(n_nearest**2)*i + n_nearest*j + k, :, 0] += granularity*(i)*(-1 if negative else +1)
#                 expanded[(n_nearest**2)*i + n_nearest*j + k, :, 1] += granularity*(j)*(-1 if negative else +1)
#                 expanded[(n_nearest**2)*i + n_nearest*j + k, :, 2] += granularity*(k)*(-1 if negative else +1)
    
#     expanded = torch.clamp(expanded, min=torch.tensor([0,0,0], dtype=int), max=torch.tensor(nwires, dtype=int))
#     # Now, we ned to get this in terms of the crossing indices.
#     # check_A_in_B looks for the pairs we created in "expanded"
#     # and outputs (index within A, index within B).
#     #
#     # So it ties the pair in question (index in A) to the +-nth crossing (index in B)
#     #
#     # Outside of this, we account for the fact that we concatenate all plane 
#     # pairs into one large crossing tensor
#     nearest_neighbors_01 = torch.cat([
#         check_A_in_B(expanded[i], indices) for i in range(expanded.shape[0])
#     ], dim=1)
#     return nearest_neighbors_01

# def make_random_neighbors(high, n=int(1775227/2)):
#     return torch.randint(
#                         low=0, 
#                         high=high,
#                         size=(2, n),
#                         dtype=torch.long
#                     )

# def create_permutation_tensor(ni: int, nj: int, nk: int) -> torch.Tensor:
#     """
#     Generates an N x 3 tensor containing all permutations (i, j, k) where
#     N = ni * nj * nk, using the optimized torch.cartesian_prod function.

#     The indices start from 0 and go up to (n-1) for each dimension.

#     Args:
#         ni: The size of the first dimension (i-index range: 0 to ni-1).
#         nj: The size of the second dimension (j-index range: 0 to nj-1).
#         nk: The size of the third dimension (k-index range: 0 to nk-1).

#     Returns:
#         A PyTorch tensor (N x 3) containing all unique (i, j, k) combinations.
#     """
#     # 1. Create 1D tensors of indices for each dimension
#     i_indices = torch.arange(ni, dtype=torch.long)
#     j_indices = torch.arange(nj, dtype=torch.long)
#     k_indices = torch.arange(nk, dtype=torch.long)

#     # 2. Use torch.cartesian_prod to compute the Cartesian product (all permutations)
#     # This is the canonical PyTorch method for this operation.
#     # The output format is automatically the desired (N, 3) shape.
#     permutation_tensor = torch.cartesian_prod(i_indices, j_indices, k_indices)

#     return permutation_tensor


import math

class Network(nn.Module):
    # @torch.autocast('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16) #TODO -- Fix this
    def __init__(
            self,
            wires_file='protodunevd-wires-larsoft-v3.json.bz2',
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
        self.unets = nn.ModuleList([
                UNet(n_channels=n_input_features, n_classes=1,
                    batch_norm=True, bilinear=True, padding=True)
                for i in range(3)
        ])

        self.split_gpu = True
        self.do_unets_two_times = True
        self.unet2_device = 'cuda:1' if (self.split_gpu and self.do_unets_two_times) else 'cuda:0'
        if self.do_unets_two_times:
            self.unets2 = nn.ModuleList([
                    UNet(n_channels=4, n_classes=1,
                        batch_norm=True, bilinear=True, padding=True)
                    for i in range(3)
            ])
        # self.skip_GNN=skip_GNN
        # self.one_side=one_side
        # if skip_unets:
        #     n_unet_features=n_input_features

        # self.features_into_GNN = 3*(n_unet_features + n_feat_wire) + 1# + 8
        # single_layer_UGNN = True
        
        # self.scramble=False
        # self.random_neighbors=False
        # self.nrandom_neighbors = int(1775227/2) #100
        # self.nscramble = 100000 #150000

        # if (not single_layer_UGNN) and (self.random_neighbors or self.scramble):
        #     raise Exception('Cannot do multi-layer UGNN and either random neighbors or scramble')
        
        # self.skip_area = (False or self.scramble)
        # self.skip_edge_attr = (False or self.scramble)

        # if self.skip_area: self.features_into_GNN -= 1
        
        
        
        # if single_layer_UGNN:
        #     encoding_message_passes = [4]
        #     encoding_hidden_chans = [16]
        #     encoding_output_chans = [8]

        #     decoding_message_passes = []
        #     decoding_hidden_chans = []
        #     decoding_output_chans = []
        #     self.runs = []
        # else:
        #     encoding_message_passes = [4, 4, 4]
        #     encoding_hidden_chans = [16, 16, 32]
        #     encoding_output_chans = [16, 32, 64]
        #     self.runs = [3, 9]
            
        #     # decoding_message_passes = [4]
        #     # decoding_hidden_chans = [16]
        #     # decoding_output_chans = [16]

        #     #Choose to make this symmetric?
        #     decoding_message_passes = encoding_message_passes[-2::-1]
        #     decoding_hidden_chans = encoding_hidden_chans[-2::-1]
        #     decoding_output_chans = encoding_output_chans[-2::-1]

        # decoding_input_chans = [sum(encoding_output_chans[-2:])]
        # for i in range(1, len(encoding_output_chans)-1):
        #     decoding_input_chans.append(decoding_output_chans[i-1] + encoding_output_chans[-(2+i)])

        # self.UGNN_encoding = nn.ModuleList([
        #     GAT(
        #         self.features_into_GNN if i == 0 else encoding_output_chans[i-1],
        #         encoding_hidden_chans[i],
        #         encoding_message_passes[i],
        #         out_channels=encoding_output_chans[i]
        #     ) for i in range(len(encoding_message_passes))
        # ])
        # self.UGNN_decoding = nn.ModuleList([
        #     GAT(
        #         decoding_input_chans[i],
        #         decoding_hidden_chans[i],
        #         decoding_message_passes[i],
        #         out_channels=decoding_output_chans[i]
        #     ) for i in range(len(decoding_message_passes))
        # ])

        # print(self.UGNN_encoding)
        # print(self.UGNN_decoding)

        # self.out_channels=encoding_output_chans[0]
        self.mlp = nn.Linear((n_unet_features), 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
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
            self.coords_face1 = xover.coords_from_schema(store, 1)
            
            # if not self.scramble:
            print('Making cells face 0')
            self.good_indices_0 = xover.make_cells(self.coords_face0, *(self.nwires_0), keep_shape=False)
            print('Done', self.good_indices_0.shape)
            print('Making cells face 1')
            self.good_indices_1 = xover.make_cells(self.coords_face1, *(self.nwires_1), keep_shape=False)
            print('Done', self.good_indices_1.shape)

            # #For UGNN inputs
            # self.UGNN_blobs_0 = [self.good_indices_0] #These are used to get geom info (area/centroids)
            # self.UGNN_blobs_1 = [self.good_indices_1]

            # self.UGNN_indices_0 = [] #These are used to merge/split info during down/up in U GNN
            # self.UGNN_indices_1 = []
            # self.UGNN_indices = []
            # self.UGNN_merged_crossings_0 = []
            # self.UGNN_merged_crossings_1 = []

            # inside_crossings_0 = xover.get_inside_crossings(self.coords_face0, self.good_indices_0)
            # self.UGNN_merged_crossings_0 = [xover.merge_crossings(self.coords_face0, inside_crossings_0, verbose=True)]
            
            # inside_crossings_1 = xover.get_inside_crossings(self.coords_face1, self.good_indices_1)
            # self.UGNN_merged_crossings_1 = [xover.merge_crossings(self.coords_face1, inside_crossings_1, verbose=True)]
            
            # #For now, just get without 'fixed' method -- which might not even be necessary
            # n_nearest = 2
            # if not self.random_neighbors:
            #     UGNN_neighbors_0 = get_nn_from_plane_triplet(self.good_indices_0[:, 2:, 0], self.nwires_0, n_nearest=n_nearest)
            #     UGNN_neighbors_1 = get_nn_from_plane_triplet(self.good_indices_1[:, 2:, 0], self.nwires_1, n_nearest=n_nearest)
            # else:
            #     # UGNN_neighbors_0 = torch.randint(
            #     #     low=0, 
            #     #     high=self.UGNN_blobs_0[0].shape[0],
            #     #     size=(3000000, 2), 
            #     #     dtype=torch.long
            #     # )
            #     UGNN_neighbors_0 = make_random_neighbors(self.UGNN_blobs_0[0].shape[0], self.nrandom_neighbors)
            #     # UGNN_neighbors_1 = torch.randint(
            #     #     low=0, 
            #     #     high=self.UGNN_blobs_1[0].shape[0],
            #     #     size=(3000000, 2), 
            #     #     dtype=torch.long
            #     # )
            #     UGNN_neighbors_1 = make_random_neighbors(self.UGNN_blobs_1[0].shape[0], self.nrandom_neighbors)
            # self.UGNN_neighbors = [
            #     torch.cat([
            #             UGNN_neighbors_0,
            #             UGNN_neighbors_1+len(self.UGNN_blobs_0[0])
            #         ], dim=1)
            # ]
            
            # self.nstatic_edge_attr = 7
            # static_edges_0 = self.make_edge_attr_new(
            #     UGNN_neighbors_0, None, self.nstatic_edge_attr,
            #     self.UGNN_merged_crossings_0[0], self.nwires_0, 0
            # )
            # static_edges_1 = self.make_edge_attr_new(
            #     UGNN_neighbors_1, None, self.nstatic_edge_attr,
            #     self.UGNN_merged_crossings_1[0], self.nwires_1, 0
            # )
            # self.UGNN_static_edges = [
            #     torch.cat([static_edges_0, static_edges_1])
            # # ]
            # print()
            # print('Static edges', self.UGNN_static_edges[0].shape)

            # for i, run in enumerate(self.runs):
            #     input_0 = self.UGNN_blobs_0[-1]
            #     input_1 = self.UGNN_blobs_1[-1]
            #     blobs, inds = xover.downsample_blobs(input_0, to_run=run)
            #     self.UGNN_blobs_0.append(blobs)
            #     self.UGNN_indices_0.append(inds)
            #     blobs, inds = xover.downsample_blobs(input_1, to_run=run)
            #     self.UGNN_blobs_1.append(blobs)
            #     self.UGNN_indices_1.append(inds)
                
            #     print(len(self.UGNN_blobs_0[-1]))
            #     self.UGNN_indices.append(torch.cat([
            #         self.UGNN_indices_0[-1],
            #         self.UGNN_indices_1[-1] + len(self.UGNN_blobs_0[-1]),
            #     ]))

            #     inside_crossings_0 = xover.get_inside_crossings(self.coords_face0, self.UGNN_blobs_0[-1])
            #     self.UGNN_merged_crossings_0.append(xover.merge_crossings(self.coords_face0, inside_crossings_0, verbose=True))
            #     inside_crossings_1 = xover.get_inside_crossings(self.coords_face1, self.UGNN_blobs_1[-1])
            #     self.UGNN_merged_crossings_1.append(xover.merge_crossings(self.coords_face1, inside_crossings_1, verbose=True))

            #     UGNN_neighbors_0 = get_nn_from_plane_triplet(self.UGNN_blobs_0[-1][:, 2:, 0], self.nwires_0, n_nearest=n_nearest, granularity=run)
            #     UGNN_neighbors_1 = get_nn_from_plane_triplet(self.UGNN_blobs_1[-1][:, 2:, 0], self.nwires_1, n_nearest=n_nearest, granularity=run)

            #     self.UGNN_neighbors.append(
            #         torch.cat([
            #             UGNN_neighbors_0,
            #             UGNN_neighbors_1 + len(self.UGNN_blobs_0[-1])
            #         ], dim=1)
            #     )


            #     static_edges_0 = self.make_edge_attr_new(
            #         UGNN_neighbors_0, None, self.nstatic_edge_attr,
            #         self.UGNN_merged_crossings_0[-1], self.nwires_0, 0
            #     )
            #     static_edges_1 = self.make_edge_attr_new(
            #         UGNN_neighbors_1, None, self.nstatic_edge_attr,
            #         self.UGNN_merged_crossings_1[-1], self.nwires_1, 0
            #     )
            #     self.UGNN_static_edges.append(
            #         torch.cat([static_edges_0, static_edges_1])
            #     )
            #     print()
            #     print('Static edges', self.UGNN_static_edges[-1].shape)

            # #Account for duplicate neighbors
            # for i in range(len(self.UGNN_neighbors)):
            #     self.UGNN_neighbors[i], inds = self.UGNN_neighbors[i].T.unique(dim=0, return_inverse=True)
            #     temp_static_edges = torch.zeros((self.UGNN_neighbors[i].shape[0],self.UGNN_static_edges[i].shape[1]))
            #     print(temp_static_edges.shape, inds.shape, self.UGNN_static_edges[i].shape)
            #     inds = inds.unsqueeze(1).repeat(1, self.UGNN_static_edges[i].shape[1])
            #     self.UGNN_static_edges[i] = temp_static_edges.scatter_reduce(0, inds, self.UGNN_static_edges[i], reduce='mean', include_self=False)
            #     # test = temp_static_edges.scatter_reduce_(0, inds, self.UGNN_static_edges[i], reduce='mean', include_self=False)
            #     # self.UGNN_static_edges[i] = torch_scatter.scatter_mean(self.UGNN_static_edges[i], inds, dim=0)
            #     # print(torch.all(test == self.UGNN_static_edges[i]))
            #     self.UGNN_neighbors[i] = self.UGNN_neighbors[i].T

            #Get areas and centers
            # print('Getting areas and centroids -- Face 0')
            # # inside_crossings_0 = xover.get_inside_crossings(self.coords_face0, self.good_indices_0)
            # # self.merged_crossings_0 = xover.merge_crossings(self.coords_face0, inside_crossings_0, verbose=True)
            # self.merged_crossings_0 = self.UGNN_merged_crossings_0[0]            
            # print()

            # print('Done')
            # print('Getting areas and centroids -- Face 1')
            # # inside_crossings_1 = xover.get_inside_crossings(self.coords_face1, self.good_indices_1)
            # # self.merged_crossings_1 = xover.merge_crossings(self.coords_face1, inside_crossings_1, verbose=True)
            # self.merged_crossings_1 = self.UGNN_merged_crossings_1[0]
            # print()
            # print('Done')

            # self.good_indices_0 = self.good_indices_0[:, 2:, 0]
            # self.good_indices_1 = self.good_indices_1[:, 2:, 0]
            # # else:
            #     self.good_indices_0 = create_permutation_tensor(*self.nwires_0)
            #     self.good_indices_0 = self.good_indices_0[torch.randperm(self.good_indices_0.shape[0])[:self.nscramble]]

            #     self.good_indices_1 = create_permutation_tensor(*self.nwires_1)
            #     self.good_indices_1 = self.good_indices_1[torch.randperm(self.good_indices_1.shape[0])[:self.nscramble]]

            #     #For UGNN inputs
            #     self.UGNN_blobs_0 = [self.good_indices_0] #These are used to get geom info (area/centroids)
            #     self.UGNN_blobs_1 = [self.good_indices_1]

            #     neighbors_0 = make_random_neighbors(self.UGNN_blobs_0[0].shape[0], self.nrandom_neighbors)
            #     neighbors_1 = make_random_neighbors(self.UGNN_blobs_1[0].shape[0], self.nrandom_neighbors)
            #     self.UGNN_neighbors = [
            #         torch.cat([
            #             neighbors_0,
            #             neighbors_1+len(self.UGNN_blobs_0[0])
            #         ], dim=1)
            #     ]
            #     self.merged_crossings_0 = None
            #     self.merged_crossings_1 = None
            #     print(self.UGNN_blobs_0)

            # view_base = len(self.coords_face0.views) - 3
            # ##These might have become irrelevant due to how we're now getting areas/centroids
            # ray_crossings_0_01 = self.coords_face0.ray_crossing(view_base + 0, self.good_indices_0[:,0], view_base + 1, self.good_indices_0[:,1])
            # ray_crossings_0_12 = self.coords_face0.ray_crossing(view_base + 1, self.good_indices_0[:,1], view_base + 2, self.good_indices_0[:,2])
            # ray_crossings_0_20 = self.coords_face0.ray_crossing(view_base + 2, self.good_indices_0[:,2], view_base + 0, self.good_indices_0[:,0])
            # ray_crossings_1_01 = self.coords_face1.ray_crossing(view_base + 0, self.good_indices_1[:,0], view_base + 1, self.good_indices_1[:,1])
            # ray_crossings_1_12 = self.coords_face1.ray_crossing(view_base + 1, self.good_indices_1[:,1], view_base + 2, self.good_indices_1[:,2])
            # ray_crossings_1_20 = self.coords_face1.ray_crossing(view_base + 2, self.good_indices_1[:,2], view_base + 0, self.good_indices_1[:,0])

            # self.ray_crossings_0 = torch.cat(
            #     [ray_crossings_0_01.unsqueeze(1), ray_crossings_0_12.unsqueeze(1), ray_crossings_0_20.unsqueeze(1)],
            #     dim=1
            # )
            # self.ray_crossings_1 = torch.cat(
            #     [ray_crossings_1_01.unsqueeze(1), ray_crossings_1_12.unsqueeze(1), ray_crossings_1_20.unsqueeze(1)],
            #     dim=1
            # )

            # self.ray_crossings_0 /= torch.norm(self.coords_face0.bounding_box, dim=1)
            # self.ray_crossings_1 /= torch.norm(self.coords_face1.bounding_box, dim=1)
            # self.ray_crossings_0 = None
            # self.ray_crossings_1 = None
        

            self.nchans = [476, 476, 292, 292]

            self.save = True
    # def get_connected(self, plane, cells_src, cells_target, src_wire_chans, target_wire_chans):
    #     results = []
    #     #Testing: build up face 0 plane 0
    #     # checked_channels = []
    #     for wire_chan in src_wire_chans:
    #         channel = wire_chan[1]
    #         # if channel in checked_channels: continue
    #         wire = wire_chan[0]
    #         src_indices = torch.where(cells_src[:,plane] == wire)
    #         # print('Checking', wire_chan)

    #         # matched_src = torch.where(src_wire_chans[:,1] == channel)
    #         # print('Mached', matched)
    #         # connected_wires_src = src_wire_chans[matched_src][:,0]
    #         # src_indices = torch.where(cells_src[:,plane] == connected_wires_src)
    #         # print('Found connected wires', connected_wires)

    #         # if connected_wires_src.size(0) == 0:
    #             # print('Skipping')
    #             # continue
    #         matched_target = torch.where(target_wire_chans[:,1] == channel)
    #         connected_wires_target = target_wire_chans[matched_target][:,0]
    #         if connected_wires_target.size(0) == 0:
    #             continue
    #         # target_indices = torch.where(cells_target[:, plane] == )
    #         # print(connected_wires_target, cells_target[:, plane])
    #         # target_indices = check_A_in_B(connected_wires_target.unsqueeze(0), cells_target[:, plane].unsqueeze(0))
    #         target_indices = []
    #         for target_wire in connected_wires_target:
    #             # print(target_wire)
    #             these_target_indices = torch.where(cells_target[:, plane] == target_wire)
    #             # print(these_target_indices)
    #             # print(cells_target[these_target_indices])
    #             target_indices.append(these_target_indices[0])
    #         target_indices = torch.cat(target_indices)
    #         # print(target_indices)
    #         # torch.where(cells_target[:,plane] == connected_wires_src)
    #         # print(
    #         #     cells_target[target_indices]
    #         # )
    #         # print('They will match to our src indices')
    #         # print(src_indices)
    #         crossed = xover.build_cross(src_indices[0], target_indices)
    #         # print(crossed.shape)
    #         # print(crossed)
    #         results.append(crossed.T)
    #         # checked_channels.append(channel)
    #     return torch.cat(results,dim=1)

        # torch.where(cells[:,0] == chanmap[0,0][torch.where(chanmap[(0,0)][:,1] == 200)][:,0])

    def fill_window_mp(self, first_wires, second_wires, third_wires, indices, type='mp3'):
        w = second_wires*third_wires

        if type == 'mp3':
            w = w*first_wires
        elif type == 'mp2':
            w = 1 - w*first_wires
        return w
            
    def fill_window(self, w, nfeat, first_wires, second_wires, third_wires, indices, crossings, merged_crossings):
        
        w[..., :(nfeat)] = first_wires[:, :, indices[:,0], :]
        w[..., (nfeat):2*(nfeat)] = second_wires[:, :, indices[:,1], :]
        w[..., 2*(nfeat):3*(nfeat)] = third_wires[:, :, indices[:,2], :]
        start = 3*(nfeat)

        if not self.skip_area:
            w[..., start] = merged_crossings['areas']
            start += 1
        # w[..., start:start+2] = crossings[:, 0].view(1, 1, -1, 2).repeat(w.shape[0], w.shape[1], 1, 1).to(w.device)
        # start += 2
        # w[..., start:start+2] = crossings[:, 1].view(1, 1, -1, 2).repeat(w.shape[0], w.shape[1], 1, 1).to(w.device)
        # start += 2
        # w[..., start:start+2] = crossings[:, 2].view(1, 1, -1, 2).repeat(w.shape[0], w.shape[1], 1, 1).to(w.device)
        # start += 2
        # w[..., start:start+2] = torch.mean(crossings, dim=1).view(1,1,-1,2).repeat(w.shape[0], w.shape[1], 1, 1).to(w.device)
    
    # def make_edge_attr(self, neighbors, cells, nattr, crossings, nwires, dface=0):
    #     edge_attrs = torch.zeros(neighbors.size(1), nattr)
    #     edge_attrs[:, :2] = (
    #         crossings[neighbors[0], 0] -
    #         crossings[neighbors[1], 0]
    #     ) #dZ, dY ij crossing
    #     edge_attrs[:, 2:4] = (
    #         crossings[neighbors[0], 1] -
    #         crossings[neighbors[1], 1]
    #     ) #dZ, dY jk crossing
    #     edge_attrs[:, 4:6] = (
    #         crossings[neighbors[0], 1] -
    #         crossings[neighbors[1], 1]
    #     ) #dZ, dY ki crossing
    #     edge_attrs[:, 6] = torch.norm(edge_attrs[:, :2], dim=1) # r for ij
    #     edge_attrs[:, 7] = torch.norm(edge_attrs[:, 2:4], dim=1) # r for jk
    #     edge_attrs[:, 8] = torch.norm(edge_attrs[:, 4:6], dim=1) # r for ki

    #     edge_attrs[:, 9:12] = cells[neighbors[0]] - cells[neighbors[1]] / torch.Tensor(nwires)

    #     edge_attrs[:, -1] = dface #dFace
    #     return edge_attrs.detach()

    # def make_edge_attr_new(self, neighbors, cells, nattr, merged_crossings, nwires, dface=0):
    #     if cells is None: nattr -= 3
    #     edge_attrs = torch.zeros(neighbors.size(1), nattr)
    #     centroids = merged_crossings['centroids']
    #     edge_attrs[:, :2] = (
    #         centroids[neighbors[0]] -
    #         centroids[neighbors[1]]
    #     )

    #     if cells is not None:
    #         edge_attrs[:, 3:6] = cells[neighbors[0]] - cells[neighbors[1]] / torch.Tensor(nwires)

    #     edge_attrs[:, -1] = dface #dFace
    #     return edge_attrs.detach()

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
        print(wire_chans)
        # print('Input', x.shape[0], len(wire_chans), x.shape[-1])
        wires = torch.zeros((x.shape[0], x.shape[1], len(wire_chans), x.shape[-1])).to(x.device)
        # print('Wires')
        # print(x.shape, wires.shape, torch.max(wire_chans[:, 0]), torch.max(wire_chans[:, 1]))
        wires[..., wire_chans[:, 0], :] = x[..., wire_chans[:,1], :]
        # wires[..., x.shape[-1]] = plane
        # wires[..., x.shape[-1]+1] = face
        return wires

    # def checkpointed_gnn_call(self, gnn, window, neighbors, edge_attr):
    #     return gnn(window, neighbors, edge_attr=edge_attr)

    # def ugnn_method(self, window, neighbors, edge_attr):
    #     '''
    #     '''
    #     outputs = []
    #     for i, encoder in enumerate(self.UGNN_encoding):
    #         #Pass through encoding step
    #         # print('Window:', window.shape)
    #         if i == 0:
    #             input = window.reshape(-1, window.shape[-1])
    #         else:
    #             #Downsample the input
    #             # input = torch.cat([
    #             #     torch_scatter.scatter_mean(outputs[-1][:len(self.UGNN_indices_0[i])], self.UGNN_indices_0[i].to(output[-1].device), dim=1),
    #             #     torch_scatter.scatter_mean(outputs[-1][len(self.UGNN_indices_0[i]):], self.UGNN_indices_1[i].to(output[-1].device), dim=1)
    #             # ], dim=1)
    #             # print(outputs[-1].shape)

    #             prev_len = outputs[-1].shape[0]

    #             #Target length per time window --> number of blobs in this layer/level
    #             target_len = len(self.UGNN_blobs_0[i]) + len(self.UGNN_blobs_1[i])
                
    #             nindices = len(self.UGNN_indices[i-1])
    #             # print(nindices, prev_len, target_len)
    #             time_window_indices = torch.zeros((self.time_window*nindices),dtype=torch.long)
    #             for j in range(self.time_window):
    #                 time_window_indices[j*nindices:(j+1)*nindices] = (self.UGNN_indices[i-1] + j*target_len)
    #             device = outputs[-1].device
    #             # input = torch_scatter.scatter_mean(outputs[-1], time_window_indices.to(device), dim=0)
    #             input = torch.zeros((target_len*self.time_window, outputs[-1].shape[1]), device=device)
    #             time_window_indices = time_window_indices.unsqueeze(1).repeat(1, outputs[-1].shape[1])
    #             input = input.scatter_reduce(0, time_window_indices.to(device), outputs[-1], reduce='mean', include_self=False)
    #             # print(test.dtype, input.dtype)
    #             # print(test == input)
    #             # print(torch.max(torch.abs(test - input)))
    #             # print(input.shape)

    #         if self.checkpoint:
    #             output = checkpoint.checkpoint(
    #                 # self.GNN,
    #                 # window,
    #                 # all_neighbors,
    #                 # None, #-- Do we need this for the checkpointing????
    #                 # edge_attr,
    #                 self.checkpointed_gnn_call,
    #                 encoder,
    #                 input,
    #                 neighbors[i],
    #                 (None if self.skip_edge_attr else edge_attr[i]),
    #             )
    #         else:
    #             output = encoder(
    #                 input,
    #                 neighbors[i],
    #                 edge_attr=(None if self.skip_edge_attr else edge_attr[i]),
    #             )
            
            
    #         outputs.append(output)
    #     # print('Outputs!')
    #     # for op in outputs:
    #     #     print(op.shape)
        
    #     output = outputs.pop(-1)
    #     if len(self.UGNN_decoding) > 0: #Special case: a single encoding layer i.e. not a UGNN
    #         for i, decoder in enumerate(self.UGNN_decoding):

    #             full_indices = self.UGNN_indices[-(1+i)].repeat(self.time_window)
    #             nindices = self.UGNN_indices[-(1+i)].shape[0]
    #             ncells = (len(self.UGNN_blobs_0[-(1+i)]) + len(self.UGNN_blobs_1[-(1+i)]))
    #             full_indices[nindices:2*nindices] += ncells
    #             full_indices[2*nindices:3*nindices] += 2*ncells
    #             # print(len(full_indices), torch.max(full_indices))
                

    #             #This upsamples from the previous layer
    #             output = output[full_indices] ##NEED TO ACCOUNT FOR time window here!!!

    #             #This combines with the current layer
    #             output = torch.cat([
    #                 output, #upsampled
    #                 outputs.pop(-1) #'current' layer
    #             ], dim=-1)
    #             if self.checkpoint:
    #                 output = checkpoint.checkpoint(
    #                     self.checkpointed_gnn_call,
    #                     decoder,
    #                     output,
    #                     neighbors[-(1+i)],
    #                     edge_attr[-(1+i)]
    #                 )
    #             else:
    #                 output = decoder(
    #                     output,
    #                     neighbors[-(1+i)],
    #                     edge_attr=edge_attr[-(1+i)]
    #                 )
    #     return output

    def forward(self, x):
        '''
        Input data is assumed to be of shape (nbatch, nfeatures, nchannels, nticks)
        '''
        input_shape = x.shape
        nbatches = x.shape[0]
        nticks = x.shape[-1]
        nchannels = x.shape[-2]

        the_device = x.device
        if self.do_unets_two_times:
            if the_device != 'cpu':
                split_device = self.unet2_device
            else: split_device = 'cpu'
            self.unets2 = nn.ModuleList([ui.to(self.unet2_device) for ui in self.unets2])
        # print('Pre unet', x.shape)

        # if not self.skip_unets:
        if self.save:
            torch.save(x, 'input_test.pt')
        x = [
            x[:, :, (0 if i == 0 else sum(self.nchans[:i])):sum(self.nchans[:i+1]), :]
            for i, nc in enumerate(self.nchans)
        ]
        for xi in x: print(xi.shape)

        #Pass through the unets
        x = [
            (self.unets[(i if i < 3 else 2)](x[i]) if not self.checkpoint else
            checkpoint.checkpoint(
                self.unets[(i if i < 3 else 2)],
                x[i]
            )) for i in range(len(x))
        ]

        x = [self.sigmoid(xi) for xi in x]
        # x = [self.relu(xi) if self.do_unets_two_times else self.sigmoid(xi) for xi in x]

        print('passed through unets')
        for xi in x: print(xi.shape)
        if self.save:
            torch.save(x, 'post_unet_test.pt')
            # print('Post unet', x.shape)

        #Cat to get into global channel number shape
        x = torch.cat(x, dim=2)

        self.good_indices_0 = self.good_indices_0.to(the_device)
        self.good_indices_1 = self.good_indices_1.to(the_device)

        # for k, v in self.face_plane_wires_channels.items():
        #     self.face_plane_wires_channels[k] = v.to(the_device)

        x = self.calculate_crossview(x)
        if self.do_unets_two_times:
            x = x.to(split_device)
            # if the_device != 'cpu':
            #     split_device = self.unet2_device
            # else: split_device = 'cpu'
            x = [
                x[:, :, (0 if i == 0 else sum(self.nchans[:i])):sum(self.nchans[:i+1]), :]
                for i, nc in enumerate(self.nchans)
            ]
            for xi in x: print(xi.shape)

            #Pass through the unets
            # self.unets2 = nn.ModuleList([ui.to(self.unet2_device) for ui in self.unets2])
            x = [
                (self.unets2[(i if i < 3 else 2)](x[i]) if not self.checkpoint else
                checkpoint.checkpoint(
                    self.unets2[(i if i < 3 else 2)],
                    x[i]
                )) for i in range(len(x))
            ]
            x = torch.cat(x, dim=2)
            x = self.sigmoid(x)
            x = self.calculate_crossview(x) #Currently out of memory on 6090 if trying to do this 


        return x.to(the_device)
    
    def calculate_crossview(self, input):
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
            print(xi.shape)
            as_wires_f0_p0 = self.make_wires(xi, 0, 0)[:, :, self.good_indices_0[:, 0]]
            as_wires_f0_p1 = self.make_wires(xi, 0, 1)[:, :, self.good_indices_0[:, 1]]
            as_wires_f0_p2 = self.make_wires(xi, 0, 2)[:, :, self.good_indices_0[:, 2]]
            as_wires_f1_p0 = self.make_wires(xi, 1, 0)[:, :, self.good_indices_1[:, 0]]
            as_wires_f1_p1 = self.make_wires(xi, 1, 1)[:, :, self.good_indices_1[:, 1]]
            as_wires_f1_p2 = self.make_wires(xi, 1, 2)[:, :, self.good_indices_1[:, 2]]
            
            # mem0 = torch.cuda.memory_allocated(0) / (1024**2)
            
            # chans_mp3 = self.scatter_to_chans(
            #     [
            #         as_wires_f0_p0*as_wires_f0_p1*as_wires_f0_p2,
            #         as_wires_f1_p0*as_wires_f1_p1*as_wires_f1_p2
            #     ], nbatches, nchannels, the_device)
            # mem1 = torch.cuda.memory_allocated(0) / (1024**2)
            # print(f'{mem0:.2f} {mem1:.2f}')

            # chans_mp2 = self.scatter_to_chans(
            #     [
            #         (1-as_wires_f0_p0)*as_wires_f0_p1*as_wires_f0_p2,
            #         (1-as_wires_f1_p0)*as_wires_f1_p1*as_wires_f1_p2
            #     ],
            #     nbatches, nchannels, the_device
            # )

            # chans_mp2 = chans_mp2 + self.scatter_to_chans(
            #     [
            #         (1-as_wires_f0_p1)*as_wires_f0_p2*as_wires_f0_p0,
            #         (1-as_wires_f1_p1)*as_wires_f1_p2*as_wires_f1_p0
            #     ],
            #     nbatches, nchannels, the_device
            # )

            # chans_mp2 = chans_mp2 + self.scatter_to_chans(
            #     [
            #         (1-as_wires_f0_p2)*as_wires_f0_p0*as_wires_f0_p1,
            #         (1-as_wires_f1_p2)*as_wires_f1_p0*as_wires_f1_p1
            #     ],
            #     nbatches, nchannels, the_device
            # )

            # crossview_chans_mp3.append(chans_mp3)
            # crossview_chans_mp2.append(chans_mp2)

            crossview_p0 = self.scatter_to_chans(
                [as_wires_f0_p0, as_wires_f1_p0],
                nbatches, nchannels, the_device
            )
            crossview_p1 = self.scatter_to_chans(
                [as_wires_f0_p1, as_wires_f1_p1],
                nbatches, nchannels, the_device
            )
            crossview_p2 = self.scatter_to_chans(
                [as_wires_f0_p2, as_wires_f1_p2],
                nbatches, nchannels, the_device
            )

            crossview_chans.append(
                torch.cat([
                    # crossview_p0.unsqueeze(1),
                    # crossview_p1.unsqueeze(1),
                    # crossview_p2.unsqueeze(1),
                    crossview_p0,
                    crossview_p1,
                    crossview_p2,
                ], dim=1)
            )
        
        # crossview_chans_mp3 = torch.cat(crossview_chans_mp3, dim=-1)
        # crossview_chans_mp2 = torch.cat(crossview_chans_mp2, dim=-1)
        # print(crossview_chans.shape, x.shape)
        x = torch.cat([
            input,
            # crossview_chans_mp3.unsqueeze(1),
            # crossview_chans_mp2.unsqueeze(1),
            torch.cat(crossview_chans, dim=-1)
        ], dim=1)
        return x

    def make_label_nodes(self, labels):
        # if self.do_unets_two_times: return labels
        '''
        We get the labels in a shape of (batch, feat=1, channels, ticks)

        So we need to go from channels view to the wires then nodes view.
        '''
        # print('Labels shape', labels.shape)
        # crossview_chans = []
        # for i in range(labels.shape[-1]):
        #     xi = labels[..., i].unsqueeze(-1)
        #     as_wires_f0_p0 = self.make_wires(xi, 0, 0)[:, self.good_indices_0[:, 0]]
        #     as_wires_f0_p1 = self.make_wires(xi, 0, 1)[:, self.good_indices_0[:, 1]]
        #     as_wires_f0_p2 = self.make_wires(xi, 0, 2)[:, self.good_indices_0[:, 2]]
        #     as_wires_f1_p0 = self.make_wires(xi, 1, 0)[:, self.good_indices_1[:, 0]]
        #     as_wires_f1_p1 = self.make_wires(xi, 1, 1)[:, self.good_indices_1[:, 1]]
        #     as_wires_f1_p2 = self.make_wires(xi, 1, 2)[:, self.good_indices_1[:, 2]]

        #     node_vals_f0 = as_wires_f0_p0*as_wires_f0_p1*as_wires_f0_p2
        #     node_vals_f1 = as_wires_f1_p0*as_wires_f1_p1*as_wires_f1_p2
        #     chans = self.scatter_to_chans([node_vals_f0, node_vals_f1], labels.shape[0], labels.shape[-2], labels.device)
        #     # print(chans.shape)
        #     crossview_chans.append(chans)
        
        # crossview_chans = torch.cat(crossview_chans, dim=-1)
        # # print(crossview_chans.shape, labels.shape)
        # x = torch.cat([
        #     labels,
        #     crossview_chans.unsqueeze(1),
        # ], dim=1)

        # return x

        return self.calculate_crossview(labels)
    # def forward(self, x):
    #     outA, outA_meta = self.A(x)
    #     nregions = outA_meta['nregions']
    #     chan_results = []
    #     node_results = []
    #     for i in range(nregions):
    #         res = self.B(outA, outA_meta, i)
    #         chan_results.append(res[0].unsqueeze(-1))
    #         node_results.append(res[1].unsqueeze(-1))
        
    #     return [
    #         torch.cat(chan_results, dim=-1),
    #         torch.cat(chan_results, dim=-1)
    #     ]

    #     # return torch.cat(
    #     #     [self.B(outA, outA_meta, i)[0].unsqueeze(-1) for i in range(nregions)],
    #     #     dim=-1
    #     # )
    