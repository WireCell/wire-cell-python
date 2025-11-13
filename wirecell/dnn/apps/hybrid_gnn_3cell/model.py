import numpy as np
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch_geometric.data import Data
from torch_geometric.nn import GAT, GCN
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

def fill_window(w, nfeat, first_wires, second_wires, third_wires, indices, crossings, merged_crossings):
    
    w[..., :(nfeat)] = first_wires[:, :, indices[:,0], :]
    w[..., (nfeat):2*(nfeat)] = second_wires[:, :, indices[:,1], :]
    w[..., 2*(nfeat):3*(nfeat)] = third_wires[:, :, indices[:,2], :]
    start = 3*(nfeat)

    w[..., start] = merged_crossings['areas']
    start += 1
    # w[..., start:start+2] = crossings[:, 0].view(1, 1, -1, 2).repeat(w.shape[0], w.shape[1], 1, 1).to(w.device)
    # start += 2
    # w[..., start:start+2] = crossings[:, 1].view(1, 1, -1, 2).repeat(w.shape[0], w.shape[1], 1, 1).to(w.device)
    # start += 2
    # w[..., start:start+2] = crossings[:, 2].view(1, 1, -1, 2).repeat(w.shape[0], w.shape[1], 1, 1).to(w.device)
    # start += 2
    # w[..., start:start+2] = torch.mean(crossings, dim=1).view(1,1,-1,2).repeat(w.shape[0], w.shape[1], 1, 1).to(w.device)
    

def get_nn_from_plane_triplet(indices, nwires, n_nearest=3, granularity=1):

    #We are creating +- n_nearest neighbors --> 2 for each plane in the pair -> 6
    expanded = indices.unsqueeze(0).repeat(6*n_nearest, 1, 1)

    #Get the +- n nearest neighbors along each wire
    for i in range(n_nearest):
        expanded[i, :, 0] += (i+1)*granularity
        expanded[n_nearest+i, :, 1] += (i+1)*granularity
        expanded[2*n_nearest+i, :, 2] += (i+1)*granularity
        
        expanded[3*n_nearest+i, :, 0] -= (i+1)*granularity
        expanded[4*n_nearest+i, :, 1] -= (i+1)*granularity
        expanded[5*n_nearest+i, :, 2] -= (i+1)*granularity

    expanded = torch.clamp(expanded, min=torch.tensor([0,0,0], dtype=int), max=torch.tensor(nwires, dtype=int))

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


def get_nn_from_plane_triplet_fixed(indices, nwires, negative=False, n_nearest=3, granularity=1):

    #We are creating +- n_nearest neighbors --> 2 for each plane in the pair -> 6
    expanded = indices.unsqueeze(0).repeat(n_nearest**3, 1, 1)

    #Get the +- n nearest neighbors along each wire
    for i in range(n_nearest):
        for j in range(n_nearest):
            for k in range(n_nearest):
                expanded[(n_nearest**2)*i + n_nearest*j + k, :, 0] += granularity*(i)*(-1 if negative else +1)
                expanded[(n_nearest**2)*i + n_nearest*j + k, :, 1] += granularity*(j)*(-1 if negative else +1)
                expanded[(n_nearest**2)*i + n_nearest*j + k, :, 2] += granularity*(k)*(-1 if negative else +1)
    
    expanded = torch.clamp(expanded, min=torch.tensor([0,0,0], dtype=int), max=torch.tensor(nwires, dtype=int))
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




import math

class Network(nn.Module):
    # @torch.autocast('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16) #TODO -- Fix this
    def __init__(
            self,
            wires_file='protodunevd-wires-larsoft-v3.json.bz2',
            n_unet_features=4,
            time_window=3,
            checkpoint=False,
            n_feat_wire = 0,
            detector=0,
            n_input_features=1,
            skip_unets=False,
            skip_GNN=False,
            one_side=False,
            out_channels=16,
            use_cells=True,
            fixed_neighbors=True,
            #gcn=False, #Currently not working
        ):
        super().__init__()
        self.nfeat_post_unet=n_unet_features
        self.n_feat_wire = n_feat_wire
        self.do_ugnn = True
        self.checkpoint=checkpoint
        self.n_input_features=n_input_features
        self.skip_unets=skip_unets
        ##Set up the UNets
        self.unets = nn.ModuleList([
                UNet(n_channels=n_input_features, n_classes=n_unet_features,
                    batch_norm=True, bilinear=True, padding=True)
                for i in range(3)
        ])
        self.skip_GNN=skip_GNN
        self.one_side=one_side
        if skip_unets:
            n_unet_features=n_input_features
        self.features_into_GNN = 3*(n_unet_features + n_feat_wire) + 1# + 8
        gnn_settings = [self.features_into_GNN, 16, 4]
        self.GNN = (
            # GCN(*gnn_settings, out_channels=out_channels) if gcn else #Currently not working
            GAT(*gnn_settings, out_channels=out_channels)
        )

        single_layer_UGNN = False
        if single_layer_UGNN:
            ugnn_message_passes = [4]
            ugnn_hidden_chans = [16]
            ugnn_output_chans = [16]

            decoding_message_passes = []
            decoding_hidden_chans = []
            decoding_output_chans = []
            self.runs = []
        else:
            encoding_message_passes = [4, 4, 4]
            encoding_hidden_chans = [16, 16, 32]
            encoding_output_chans = [16, 32, 64]
            self.runs = [3, 9]
            
            # decoding_message_passes = [4]
            # decoding_hidden_chans = [16]
            # decoding_output_chans = [16]

            #Choose to make this symmetric?
            decoding_message_passes = encoding_message_passes[-2::-1]
            decoding_hidden_chans = encoding_hidden_chans[-2::-1]
            decoding_output_chans = encoding_output_chans[-2::-1]

        decoding_input_chans = [sum(encoding_output_chans[-2:])]
        for i in range(1, len(encoding_output_chans)-1):
            decoding_input_chans.append(decoding_output_chans[i-1] + encoding_output_chans[-(2+i)])

        self.UGNN_encoding = nn.ModuleList([
            GAT(
                self.features_into_GNN if i == 0 else encoding_output_chans[i-1],
                encoding_hidden_chans[i],
                encoding_message_passes[i],
                out_channels=encoding_output_chans[i]
            ) for i in range(len(encoding_message_passes))
        ])
        self.UGNN_decoding = nn.ModuleList([
            GAT(
                decoding_input_chans[i],
                decoding_hidden_chans[i],
                decoding_message_passes[i],
                out_channels=decoding_output_chans[i]
            ) for i in range(len(decoding_message_passes))
        ])

        print(self.UGNN_encoding)
        print(self.UGNN_decoding)

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
            
            # if use_cells:
            print('Making cells face 0')
            self.good_indices_0 = xover.make_cells(self.coords_face0, *(self.nwires_0), keep_shape=True)
            print('Done', self.good_indices_0.shape)
            print('Making cells face 1')
            self.good_indices_1 = xover.make_cells(self.coords_face1, *(self.nwires_1), keep_shape=True)
            print('Done', self.good_indices_1.shape)

            #For UGNN inputs
            self.UGNN_blobs_0 = [self.good_indices_0] #These are used to get geom info (area/centroids)
            self.UGNN_blobs_1 = [self.good_indices_1]

            self.UGNN_indices_0 = [] #These are used to merge/split info during down/up in U GNN
            self.UGNN_indices_1 = []
            self.UGNN_indices = []
            self.UGNN_merged_crossings_0 = []
            self.UGNN_merged_crossings_1 = []

            inside_crossings_0 = xover.get_inside_crossings(self.coords_face0, self.good_indices_0)
            self.UGNN_merged_crossings_0 = [xover.merge_crossings(self.coords_face0, inside_crossings_0, verbose=True)]
            
            inside_crossings_1 = xover.get_inside_crossings(self.coords_face1, self.good_indices_1)
            self.UGNN_merged_crossings_1 = [xover.merge_crossings(self.coords_face1, inside_crossings_1, verbose=True)]
            
            #For now, just get without 'fixed' method -- which might not even be necessary
            n_nearest = 2
            UGNN_neighbors_0 = get_nn_from_plane_triplet(self.good_indices_0[:, 2:, 0], self.nwires_0, n_nearest=n_nearest)
            UGNN_neighbors_1 = get_nn_from_plane_triplet(self.good_indices_1[:, 2:, 0], self.nwires_1, n_nearest=n_nearest)
            self.UGNN_neighbors = [
                torch.cat([
                        UGNN_neighbors_0,
                        UGNN_neighbors_1+len(self.UGNN_blobs_0[0])
                    ], dim=1)
            ]
            
            self.nstatic_edge_attr = 7
            static_edges_0 = self.make_edge_attr_new(
                UGNN_neighbors_0, None, self.nstatic_edge_attr,
                self.UGNN_merged_crossings_0[0], self.nwires_0, 0
            )
            static_edges_1 = self.make_edge_attr_new(
                UGNN_neighbors_1, None, self.nstatic_edge_attr,
                self.UGNN_merged_crossings_1[0], self.nwires_1, 0
            )
            self.UGNN_static_edges = [
                torch.cat([static_edges_0, static_edges_1])
            ]
            print()
            print('Static edges', self.UGNN_static_edges[0].shape)

            for i, run in enumerate(self.runs):
                input_0 = self.UGNN_blobs_0[-1]
                input_1 = self.UGNN_blobs_1[-1]
                blobs, inds = xover.downsample_blobs(input_0, to_run=run)
                self.UGNN_blobs_0.append(blobs)
                self.UGNN_indices_0.append(inds)
                blobs, inds = xover.downsample_blobs(input_1, to_run=run)
                self.UGNN_blobs_1.append(blobs)
                self.UGNN_indices_1.append(inds)
                
                print(len(self.UGNN_blobs_0[-1]))
                self.UGNN_indices.append(torch.cat([
                    self.UGNN_indices_0[-1],
                    self.UGNN_indices_1[-1] + len(self.UGNN_blobs_0[-1]),
                ]))

                inside_crossings_0 = xover.get_inside_crossings(self.coords_face0, self.UGNN_blobs_0[-1])
                self.UGNN_merged_crossings_0.append(xover.merge_crossings(self.coords_face0, inside_crossings_0, verbose=True))
                inside_crossings_1 = xover.get_inside_crossings(self.coords_face1, self.UGNN_blobs_1[-1])
                self.UGNN_merged_crossings_1.append(xover.merge_crossings(self.coords_face1, inside_crossings_1, verbose=True))

                UGNN_neighbors_0 = get_nn_from_plane_triplet(self.UGNN_blobs_0[-1][:, 2:, 0], self.nwires_0, n_nearest=n_nearest, granularity=run)
                UGNN_neighbors_1 = get_nn_from_plane_triplet(self.UGNN_blobs_1[-1][:, 2:, 0], self.nwires_1, n_nearest=n_nearest, granularity=run)

                self.UGNN_neighbors.append(
                    torch.cat([
                        UGNN_neighbors_0,
                        UGNN_neighbors_1 + len(self.UGNN_blobs_0[-1])
                    ], dim=1)
                )

                
                static_edges_0 = self.make_edge_attr_new(
                    UGNN_neighbors_0, None, self.nstatic_edge_attr,
                    self.UGNN_merged_crossings_0[-1], self.nwires_0, 0
                )
                static_edges_1 = self.make_edge_attr_new(
                    UGNN_neighbors_1, None, self.nstatic_edge_attr,
                    self.UGNN_merged_crossings_1[-1], self.nwires_1, 0
                )
                self.UGNN_static_edges.append(
                    torch.cat([static_edges_0, static_edges_1])
                )
                print()
                print('Static edges', self.UGNN_static_edges[-1].shape)


            for i in range(len(self.UGNN_neighbors)):
                self.UGNN_neighbors[i], inds = self.UGNN_neighbors[i].T.unique(dim=0, return_inverse=True)
                self.UGNN_static_edges[i] = torch_scatter.scatter_mean(self.UGNN_static_edges[i], inds, dim=0)
                self.UGNN_neighbors[i] = self.UGNN_neighbors[i].T

            #Get areas and centers
            print('Getting areas and centroids -- Face 0')
            # inside_crossings_0 = xover.get_inside_crossings(self.coords_face0, self.good_indices_0)
            # self.merged_crossings_0 = xover.merge_crossings(self.coords_face0, inside_crossings_0, verbose=True)
            self.merged_crossings_0 = self.UGNN_merged_crossings_0[0]            
            print()

            print('Done')
            print('Getting areas and centroids -- Face 1')
            # inside_crossings_1 = xover.get_inside_crossings(self.coords_face1, self.good_indices_1)
            # self.merged_crossings_1 = xover.merge_crossings(self.coords_face1, inside_crossings_1, verbose=True)
            self.merged_crossings_1 = self.UGNN_merged_crossings_1[0]
            print()
            print('Done')

            self.good_indices_0 = self.good_indices_0[:, 2:, 0]
            self.good_indices_1 = self.good_indices_1[:, 2:, 0]

            view_base = len(self.coords_face0.views) - 3
            ##These might have become irrelevant due to how we're now getting areas/centroids
            ray_crossings_0_01 = self.coords_face0.ray_crossing(view_base + 0, self.good_indices_0[:,0], view_base + 1, self.good_indices_0[:,1])
            ray_crossings_0_12 = self.coords_face0.ray_crossing(view_base + 1, self.good_indices_0[:,1], view_base + 2, self.good_indices_0[:,2])
            ray_crossings_0_20 = self.coords_face0.ray_crossing(view_base + 2, self.good_indices_0[:,2], view_base + 0, self.good_indices_0[:,0])
            ray_crossings_1_01 = self.coords_face1.ray_crossing(view_base + 0, self.good_indices_1[:,0], view_base + 1, self.good_indices_1[:,1])
            ray_crossings_1_12 = self.coords_face1.ray_crossing(view_base + 1, self.good_indices_1[:,1], view_base + 2, self.good_indices_1[:,2])
            ray_crossings_1_20 = self.coords_face1.ray_crossing(view_base + 2, self.good_indices_1[:,2], view_base + 0, self.good_indices_1[:,0])

            self.ray_crossings_0 = torch.cat(
                [ray_crossings_0_01.unsqueeze(1), ray_crossings_0_12.unsqueeze(1), ray_crossings_0_20.unsqueeze(1)],
                dim=1
            )
            self.ray_crossings_1 = torch.cat(
                [ray_crossings_1_01.unsqueeze(1), ray_crossings_1_12.unsqueeze(1), ray_crossings_1_20.unsqueeze(1)],
                dim=1
            )

            self.ray_crossings_0 /= torch.norm(self.coords_face0.bounding_box, dim=1)
            self.ray_crossings_1 /= torch.norm(self.coords_face1.bounding_box, dim=1)

            # #Neighbors on either face of the anode
            n_nearest = 2

            print('Getting neighbors')
            if not fixed_neighbors:
                nearest_neighbors_0 = get_nn_from_plane_triplet(self.good_indices_0, self.nwires_0, n_nearest=n_nearest)
                nearest_neighbors_1 = get_nn_from_plane_triplet(self.good_indices_1, self.nwires_1, n_nearest=n_nearest)
            else:
                nearest_neighbors_0 = torch.cat([
                    get_nn_from_plane_triplet_fixed(self.good_indices_0, self.nwires_0, n_nearest=n_nearest),
                    get_nn_from_plane_triplet_fixed(self.good_indices_0, self.nwires_1, n_nearest=n_nearest, negative=True),
                ], dim=1)
                nearest_neighbors_1 = torch.cat([
                    get_nn_from_plane_triplet_fixed(self.good_indices_1, self.nwires_0, n_nearest=n_nearest),
                    get_nn_from_plane_triplet_fixed(self.good_indices_1, self.nwires_1, n_nearest=n_nearest, negative=True),
                ], dim=1)



            # #Neighbors between anode faces which are connected by the elec channel?
            # #TODO

            self.neighbors = torch.cat(
                [nearest_neighbors_0] +
                ([
                    nearest_neighbors_1 + len(self.good_indices_0),
                ] if not one_side else []) + [
                    # connections_0_1
                ], dim=1)
            
            

            print(f'TOTAL EDGES: {self.neighbors.size(1)}')
            # self.neighbors = self.neighbors.T.unique(dim=0).T
            

            # #Static edge attributes -- dZ, dY, r=sqrt(dZ**2 + dY**2), dFace
            # #TODO  Do things like dWire0, dWire1 make sense for things like cross-pair (i.e. 0,1 and 0,2) neighbors?
            # #      Same question for cross face (i.e. 0,1 on face 0 and 0,1 on face 1)
            # self.nstatic_edge_attr = 13
            self.nstatic_edge_attr = 7
            static_edges_0 = self.make_edge_attr_new(
                nearest_neighbors_0, self.good_indices_0, self.nstatic_edge_attr,
                self.merged_crossings_0, self.nwires_0, 0
            )
            static_edges_1 = self.make_edge_attr_new(
                nearest_neighbors_1, self.good_indices_1, self.nstatic_edge_attr,
                self.merged_crossings_1, self.nwires_1, 0
            )


            self.static_edges = torch.cat(
                [
                    static_edges_0,
                ] + ([
                    static_edges_1,
                ] if not one_side else [])
            )
            

            

            #Get the unique neighbors
            self.neighbors, inds = self.neighbors.T.unique(dim=0, return_inverse=True)
            self.static_edges = torch_scatter.scatter_mean(self.static_edges, inds, dim=0)
            self.neighbors = self.neighbors.T
            print(f'Unique EDGES: {self.neighbors.size(1)}')
            torch.save(self.neighbors, 'neighbors.pt')


            self.nchans = [476, 476, 292, 292]

            self.save = True
    def get_connected(self, plane, cells_src, cells_target, src_wire_chans, target_wire_chans):
        results = []
        #Testing: build up face 0 plane 0
        # checked_channels = []
        for wire_chan in src_wire_chans:
            channel = wire_chan[1]
            # if channel in checked_channels: continue
            wire = wire_chan[0]
            src_indices = torch.where(cells_src[:,plane] == wire)
            # print('Checking', wire_chan)

            # matched_src = torch.where(src_wire_chans[:,1] == channel)
            # print('Mached', matched)
            # connected_wires_src = src_wire_chans[matched_src][:,0]
            # src_indices = torch.where(cells_src[:,plane] == connected_wires_src)
            # print('Found connected wires', connected_wires)

            # if connected_wires_src.size(0) == 0:
                # print('Skipping')
                # continue
            matched_target = torch.where(target_wire_chans[:,1] == channel)
            connected_wires_target = target_wire_chans[matched_target][:,0]
            if connected_wires_target.size(0) == 0:
                continue
            # target_indices = torch.where(cells_target[:, plane] == )
            # print(connected_wires_target, cells_target[:, plane])
            # target_indices = check_A_in_B(connected_wires_target.unsqueeze(0), cells_target[:, plane].unsqueeze(0))
            target_indices = []
            for target_wire in connected_wires_target:
                # print(target_wire)
                these_target_indices = torch.where(cells_target[:, plane] == target_wire)
                # print(these_target_indices)
                # print(cells_target[these_target_indices])
                target_indices.append(these_target_indices[0])
            target_indices = torch.cat(target_indices)
            # print(target_indices)
            # torch.where(cells_target[:,plane] == connected_wires_src)
            # print(
            #     cells_target[target_indices]
            # )
            # print('They will match to our src indices')
            # print(src_indices)
            crossed = xover.build_cross(src_indices[0], target_indices)
            # print(crossed.shape)
            # print(crossed)
            results.append(crossed.T)
            # checked_channels.append(channel)
        return torch.cat(results,dim=1)

        # torch.where(cells[:,0] == chanmap[0,0][torch.where(chanmap[(0,0)][:,1] == 200)][:,0])
    def make_edge_attr(self, neighbors, cells, nattr, crossings, nwires, dface=0):
        edge_attrs = torch.zeros(neighbors.size(1), nattr)
        edge_attrs[:, :2] = (
            crossings[neighbors[0], 0] -
            crossings[neighbors[1], 0]
        ) #dZ, dY ij crossing
        edge_attrs[:, 2:4] = (
            crossings[neighbors[0], 1] -
            crossings[neighbors[1], 1]
        ) #dZ, dY jk crossing
        edge_attrs[:, 4:6] = (
            crossings[neighbors[0], 1] -
            crossings[neighbors[1], 1]
        ) #dZ, dY ki crossing
        edge_attrs[:, 6] = torch.norm(edge_attrs[:, :2], dim=1) # r for ij
        edge_attrs[:, 7] = torch.norm(edge_attrs[:, 2:4], dim=1) # r for jk
        edge_attrs[:, 8] = torch.norm(edge_attrs[:, 4:6], dim=1) # r for ki

        edge_attrs[:, 9:12] = cells[neighbors[0]] - cells[neighbors[1]] / torch.Tensor(nwires)

        edge_attrs[:, -1] = dface #dFace
        return edge_attrs.detach()

    def make_edge_attr_new(self, neighbors, cells, nattr, merged_crossings, nwires, dface=0):
        if cells is None: nattr -= 3
        edge_attrs = torch.zeros(neighbors.size(1), nattr)
        centroids = merged_crossings['centroids']
        edge_attrs[:, :2] = (
            centroids[neighbors[0]] -
            centroids[neighbors[1]]
        )

        if cells is not None:
            edge_attrs[:, 3:6] = cells[neighbors[0]] - cells[neighbors[1]] / torch.Tensor(nwires)

        edge_attrs[:, -1] = dface #dFace
        return edge_attrs.detach()
    def scatter_to_chans(self, y, nbatches, nchannels, the_device):
        #TODO -- check size of y etc
        temp_out = torch.zeros(nbatches, nchannels, y[0].size(-1)).to(the_device)

        to_scatter = [
            #plane0
            [y[0], self.face_plane_wires_channels[(0,0)][self.good_indices_0[:,0]][:,1]],
            
            #plane1
            [y[0], self.face_plane_wires_channels[(0,1)][self.good_indices_0[:,1]][:,1]],
            
            #plane2
            [y[0], self.face_plane_wires_channels[(0,2)][self.good_indices_0[:,2]][:,1]],
        ] + ([] if self.one_side else [
            #plane0
            [y[1], self.face_plane_wires_channels[(1,0)][self.good_indices_1[:,0]][:,1]],

            #plane1
            [y[1], self.face_plane_wires_channels[(1,1)][self.good_indices_1[:,1]][:,1]],
            
            #plane2
            [y[1], self.face_plane_wires_channels[(1,2)][self.good_indices_1[:,2]][:,1]],
        ])

        for yi, indices in to_scatter:
            # torch_scatter.scatter_add(
            # torch_scatter.scatter_max(
            torch_scatter.scatter_mean(
                yi, indices,
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
    def ugnn_method(self, window, neighbors, edge_attr):
        '''
        '''
        outputs = []
        for i, encoder in enumerate(self.UGNN_encoding):
            #Pass through encoding step
            print('Window:', window.shape)
            if i == 0:
                input = window.reshape(-1, window.shape[-1])
            else:
                #Downsample the input
                # input = torch.cat([
                #     torch_scatter.scatter_mean(outputs[-1][:len(self.UGNN_indices_0[i])], self.UGNN_indices_0[i].to(output[-1].device), dim=1),
                #     torch_scatter.scatter_mean(outputs[-1][len(self.UGNN_indices_0[i]):], self.UGNN_indices_1[i].to(output[-1].device), dim=1)
                # ], dim=1)
                print(outputs[-1].shape)

                prev_len = outputs[-1].shape[0]

                #Target length per time window --> number of blobs in this layer/level
                target_len = len(self.UGNN_blobs_0[i]) + len(self.UGNN_blobs_1[i])
                
                nindices = len(self.UGNN_indices[i-1])
                print(nindices, prev_len, target_len)
                time_window_indices = torch.zeros((self.time_window*nindices),dtype=torch.long)
                for j in range(self.time_window):
                    time_window_indices[j*nindices:(j+1)*nindices] = (self.UGNN_indices[i-1] + j*target_len)
                input = torch_scatter.scatter_mean(outputs[-1], time_window_indices.to(outputs[-1].device), dim=0)
                print(input.shape)
            output = encoder(
                input,
                neighbors[i],
                edge_attr=edge_attr[i]
            )
            
            outputs.append(output)
        print('Outputs!')
        for op in outputs:
            print(op.shape)
        
        output = outputs.pop(-1)
        if len(self.UGNN_decoding) > 0: #Special case: a single encoding layer i.e. not a UGNN
            for i, decoder in enumerate(self.UGNN_decoding):
                #This upsamples from the previous layer
                output = output[self.UGNN_indices[-(1+i)]] ##NEED TO ACCOUNT FOR time window here!!!

                #This combines with the current layer
                output = torch.cat([
                    output, #upsampled
                    outputs.pop(-1) #'current' layer
                ], dim=-1)

                output = decoder(
                    output,
                    neighbors[-(1+i)],
                    edge_attr=edge_attr[(-(1+i))]
                )
        return output

    def A(self, x):
        '''
        Input data is assumed to be of shape (nbatch, nfeatures, nchannels, nticks)
        '''
        input_shape = x.shape
        nbatches = x.shape[0]
        nticks = x.shape[-1]
        nchannels = x.shape[-2]

        the_device = x.device
        # print('Pre unet', x.shape)

        if not self.skip_unets:
            if self.save:
                torch.save(x, 'input_test.pt')
            xs = [
                x[:, :, (0 if i == 0 else sum(self.nchans[:i])):sum(self.nchans[:i+1]), :]
                for i, nc in enumerate(self.nchans)
            ]
            # for x in xs: print(x.shape)

            #Pass through the unets
            xs = [
                (self.unets[(i if i < 3 else 2)](xs[i]) if not self.checkpoint else
                checkpoint.checkpoint(
                    self.unets[(i if i < 3 else 2)],
                    xs[i]
                )) for i in range(len(xs))
            ]

            print('passed through unets')
            # for x in xs: print(x.shape)

            #Cat to get into global channel number shape
            x = torch.cat(xs, dim=2)
            if self.save:
                torch.save(x, 'post_unet_test.pt')
            # print('Post unet', x.shape)
        n_feat_base = x.shape[1]
        nticks_orig = x.size(-1)
        #batch, tick, channels, features
        to_pad = int((self.time_window-1)/2)
        x = F.pad(x, (to_pad, to_pad))
        nticks = x.size(-1)
        x = x.permute(0,3,2,1)
        #Convert from channels to wires (values duped for common elec chan)
        these_nfeats = n_feat_base + self.n_feat_wire

        for ij, tensor in self.face_plane_wires_channels.items():
            self.face_plane_wires_channels[ij] = tensor.to(the_device)
        
        # nfeat = 3*(these_nfeats) + 8
        nfeat = self.features_into_GNN
        ncells_0 = len(self.good_indices_0)
        ncells_1 = len(self.good_indices_1)
        
        ncells = ncells_0 + ncells_1
        ranges = [[0, ncells_0], [ncells_0, ncells_0+ncells_1]]

        #in-tick crossings
        dt = self.time_window
        n_window_neighbors = self.neighbors.size(1)
        new_window_neighbors_size = n_window_neighbors*dt
        all_neighbors = torch.zeros(2, new_window_neighbors_size + ncells*((dt)**2), dtype=int).to(the_device)
        all_neighbors[:, :n_window_neighbors*(dt)] = self.neighbors.repeat(1, dt).to(the_device)
        all_neighbors[:, new_window_neighbors_size:] = torch.arange(ncells).unsqueeze(0).repeat(2,(dt)**2).to(the_device)

        for i in range(dt):
            all_neighbors[:, i*n_window_neighbors:(i+1)*n_window_neighbors] += (i*ncells)

            for j in range(dt):
                all_neighbors[0, new_window_neighbors_size + (ncells*(j*(dt) + i)):new_window_neighbors_size + (ncells*(j*(dt) + (i+1)))] += i*ncells
                all_neighbors[1, new_window_neighbors_size + (ncells*(j*(dt) + i)):new_window_neighbors_size + (ncells*(j*(dt) + (i+1)))] += j*ncells

        n_edge_attr = self.nstatic_edge_attr + 1 #+1 for tick
        edge_attr = torch.zeros(all_neighbors.size(1), n_edge_attr).to(the_device)
        # #TODO -- consider batching
        edge_attr[:new_window_neighbors_size, :-1] = self.static_edges.view(self.neighbors.size(1), -1).repeat(1*(dt), 1).to(the_device)
        base = new_window_neighbors_size

        for i in range(dt):
            for j in range(dt):
                ind_0 = (base + ncells*(j*(dt) + i))
                ind_1 = ind_0 + ncells
                edge_attr[ind_0:ind_1, -1] = (i-j)

        UGNN_all_neighbors = []
        UGNN_edge_attr = []
        
        for ilayer, neighbors in enumerate(self.UGNN_neighbors):
            n_window_neighbors = neighbors.size(1)
            new_window_neighbors_size = n_window_neighbors*dt
            these_ncells = len(self.UGNN_blobs_0[ilayer]) + len(self.UGNN_blobs_1[ilayer])
            these_all_neighbors = torch.zeros(2, new_window_neighbors_size + these_ncells*((dt)**2), dtype=int).to(the_device)
            these_all_neighbors[:, :new_window_neighbors_size] = neighbors.repeat(1, dt).to(the_device)
            these_all_neighbors[:, new_window_neighbors_size:] = torch.arange(these_ncells).unsqueeze(0).repeat(2,(dt)**2).to(the_device)

            for i in range(dt):
                these_all_neighbors[:, i*n_window_neighbors:(i+1)*n_window_neighbors] += (i*these_ncells)

                for j in range(dt):
                    these_all_neighbors[0, new_window_neighbors_size + (these_ncells*(j*(dt) + i)):new_window_neighbors_size + (these_ncells*(j*(dt) + (i+1)))] += i*these_ncells
                    these_all_neighbors[1, new_window_neighbors_size + (these_ncells*(j*(dt) + i)):new_window_neighbors_size + (these_ncells*(j*(dt) + (i+1)))] += j*these_ncells

            n_edge_attr = self.UGNN_static_edges[ilayer].shape[1] + 1 #+1 for tick
            print(self.UGNN_static_edges[ilayer].shape)
            these_edge_attr = torch.zeros(these_all_neighbors.size(1), n_edge_attr).to(the_device)
            # #TODO -- consider batching
            these_edge_attr[:new_window_neighbors_size, :-1] = self.UGNN_static_edges[ilayer].view(neighbors.size(1), -1).repeat(1*(dt), 1).to(the_device)
            base = new_window_neighbors_size

            for i in range(dt):
                for j in range(dt):
                    ind_0 = (base + these_ncells*(j*(dt) + i))
                    ind_1 = ind_0 + these_ncells
                    these_edge_attr[ind_0:ind_1, -1] = (i-j)
            UGNN_all_neighbors.append(these_all_neighbors.detach())
            UGNN_edge_attr.append(these_edge_attr)

        xmeta = dict(
            input_shape=input_shape,
            nbatches=nbatches,
            nregions=nticks_orig,
            nchannels=nchannels,
            the_device=the_device,
            to_pad=to_pad,
            all_neighbors=all_neighbors.detach(),
            edge_attr=edge_attr,
            UGNN_all_neighbors=UGNN_all_neighbors,
            UGNN_edge_attr=UGNN_edge_attr,
            these_nfeats=these_nfeats,
            ncells=ncells,
            ncells_0=ncells_0,
            ncells_1=ncells_1,
            ranges=ranges,
        )
        return x, xmeta
    def B(self, x, xmeta, tick):
        # if self.skip_GNN:
        #     x = x.permute(0,3,2,1)
        #     return self.mlp(x).permute(0,3,2,1)
        # else:
        to_pad = xmeta['to_pad']
        these_nfeats = xmeta['these_nfeats']
        nbatches = xmeta['nbatches']
        nchannels = xmeta['nchannels']
        ncells = xmeta['ncells']
        nfeat = self.features_into_GNN
        the_device = xmeta['the_device']
        low = tick
        hi = low + 2*to_pad+1
        # print('Low,hi:', low, hi)
        # low = 0
        # hi = 2*to_pad + 1

        #NEW AS WIRES
        as_wires_f0_p0 = self.make_wires(x, low, hi, these_nfeats, 0, 0)
        as_wires_f0_p1 = self.make_wires(x, low, hi, these_nfeats, 0, 1)
        as_wires_f0_p2 = self.make_wires(x, low, hi, these_nfeats, 0, 2)
        as_wires_f1_p0 = self.make_wires(x, low, hi, these_nfeats, 1, 0)
        as_wires_f1_p1 = self.make_wires(x, low, hi, these_nfeats, 1, 1)
        as_wires_f1_p2 = self.make_wires(x, low, hi, these_nfeats, 1, 2)
        ######################

        dt = self.time_window
        window = torch.zeros(
            nbatches,
            dt,
            ncells,
            nfeat,
        ).to(the_device)


        cross_start = 0
        cross_end = 0
        window_infos = [
            [as_wires_f0_p0, as_wires_f0_p1, as_wires_f0_p2, self.good_indices_0, self.ray_crossings_0, self.merged_crossings_0],
        ] + ([] if self.one_side else [
            [as_wires_f1_p0, as_wires_f1_p1, as_wires_f1_p2, self.good_indices_1, self.ray_crossings_1, self.merged_crossings_1],
        ])
        for info in window_infos:
            cross_end += len(info[3])
            fill_window(
                window[..., cross_start:cross_end, :],
                these_nfeats,
                *info
            )
            cross_start = cross_end


        # y = self.GNN(window, all_neighbors, edge_attr=edge_attr)
        if self.do_ugnn:
            neighbors = xmeta['UGNN_all_neighbors']
            edge_attr = xmeta['UGNN_edge_attr']
            self.ugnn_method(window, neighbors, edge_attr)
            sys.exit()
        else:
            window = window.reshape(-1, nfeat)
            all_neighbors = xmeta['all_neighbors']
            edge_attr = xmeta['edge_attr']
            y = checkpoint.checkpoint(
                self.GNN,
                window,
                all_neighbors,
                edge_attr,
            ) if self.checkpoint else self.GNN(window, all_neighbors, edge_attr=edge_attr)


        #Just get out the middle element
        y = y.reshape(nbatches, self.time_window, -1, self.out_channels)[:, int((self.time_window-1)/2), ...]
        ranges = xmeta['ranges']
        y = [
            y[:, r0:r1, :] for r0, r1 in ranges
        ]
        

        temp_out = self.scatter_to_chans(y, nbatches, nchannels, the_device)
        
        #batch, feat, channel
        temp_out = self.mlp(temp_out).view(1,1,-1)
        
        return temp_out
        
    def forward(self, x):
        outA, outA_meta = self.A(x)
        nregions = outA_meta['nregions']
        return torch.cat(
            [self.B(outA, outA_meta, i).unsqueeze(-1) for i in range(nregions)],
            dim=-1
        )
    