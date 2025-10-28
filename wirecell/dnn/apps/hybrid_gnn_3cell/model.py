import numpy as np
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

def fill_window(w, nfeat, first_wires, second_wires, third_wires, indices, crossings):
    
    w[..., :(nfeat)] = first_wires[:, :, indices[:,0], :]
    w[..., (nfeat):2*(nfeat)] = second_wires[:, :, indices[:,1], :]
    w[..., 2*(nfeat):3*(nfeat)] = third_wires[:, :, indices[:,2], :]
    # start = 3*(nfeat)
    # w[..., start:start+2] = crossings[:, 0].view(1, 1, -1, 2).repeat(w.shape[0], w.shape[1], 1, 1).to(w.device)
    # start += 2
    # w[..., start:start+2] = crossings[:, 1].view(1, 1, -1, 2).repeat(w.shape[0], w.shape[1], 1, 1).to(w.device)
    # start += 2
    # w[..., start:start+2] = crossings[:, 2].view(1, 1, -1, 2).repeat(w.shape[0], w.shape[1], 1, 1).to(w.device)
    # start += 2
    # w[..., start:start+2] = torch.mean(crossings, dim=1).view(1,1,-1,2).repeat(w.shape[0], w.shape[1], 1, 1).to(w.device)

def get_nn_from_plane_triplet(indices, n_nearest=3):

    #We are creating +- n_nearest neighbors --> 2 for each plane in the pair -> 6
    expanded = indices.unsqueeze(0).repeat(6*n_nearest, 1, 1)

    #Get the +- n nearest neighbors along each wire
    for i in range(n_nearest):
        expanded[i, :, 0] += (i+1)
        expanded[n_nearest+i, :, 1] += (i+1)
        expanded[2*n_nearest+i, :, 2] += (i+1)
        
        expanded[3*n_nearest+i, :, 0] -= (i+1)
        expanded[4*n_nearest+i, :, 1] -= (i+1)
        expanded[5*n_nearest+i, :, 2] -= (i+1)

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
            time_window=1,
            n_feat_wire = 0,
            detector=0,
            n_input_features=1,
            skip_unets=False,
            skip_GNN=False,
            one_side=False,
            out_channels=4,
            use_cells=True,
            #gcn=False, #Currently not working
        ):
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
        self.one_side=one_side
        if skip_unets:
            n_unet_features=n_input_features
        self.features_into_GNN = 3*(n_unet_features + n_feat_wire)# + 8
        gnn_settings = [self.features_into_GNN, 8, 2]
        self.GNN = (
            # GCN(*gnn_settings, out_channels=out_channels) if gcn else #Currently not working
            GAT(*gnn_settings, out_channels=out_channels)
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
                    # wire_chans = []
                    wire_chans = torch.zeros((len(plane.wires), 2), dtype=int)
                    for wi in plane.wires:
                        wire = store.wires[wi]
                        # wire_chans.append([wire.ident, chanmap[wire.channel]]) #convert from larsoft
                        wire_chans[wire.ident, 0] = wire.ident
                        wire_chans[wire.ident, 1] = chanmap[wire.channel]
                    # self.face_plane_wires_channels[(i,jj)] = torch.tensor(wire_chans, dtype=int)
                    self.face_plane_wires_channels[(i,jj)] = torch.tensor(wire_chans, dtype=int)
                    # print('Made fpwc:', i, jj, self.face_plane_wires_channels[(i,jj)].shape)
            

            # face_to_plane_to_nwires = {
            #     i:[len(store.planes[p].wires) for j,p in enumerate(self.faces[i].planes)] for i in face_ids
            # }
            # print(face_to_plane_to_nwires)
            
            self.nwires_0 = [len(store.planes[i].wires) for i in store.faces[0].planes]
            self.nwires_1 = [len(store.planes[i].wires) for i in store.faces[1].planes]

            self.coords_face0 = xover.coords_from_schema(store, 0)
            self.coords_face1 = xover.coords_from_schema(store, 1)
            
            if use_cells:
                print('Making cells face 0')
                self.good_indices_0 = xover.make_cells(self.coords_face0, *(self.nwires_0))
                print('Done', self.good_indices_0.shape)
                print('Making cells face 1')
                self.good_indices_1 = xover.make_cells(self.coords_face1, *(self.nwires_1))
                print('Done', self.good_indices_1.shape)
            else:
                print('Building maps')
                self.good_indices_0 = xover.build_map(self.coords_face0, self.nwires_0)
                self.good_indices_1 = xover.build_map(self.coords_face1, self.nwires_1)
                print('Done')

            view_base = len(self.coords_face0.views) - 3
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

            nearest_neighbors_0 = get_nn_from_plane_triplet(self.good_indices_0, n_nearest=n_nearest)
            nearest_neighbors_1 = get_nn_from_plane_triplet(self.good_indices_1, n_nearest=n_nearest)


            # #Neighbors between anode faces which are connected by the elec channel?
            # #TODO


            self.neighbors = torch.cat(
                [nearest_neighbors_0] +
                ([
                    nearest_neighbors_1 + len(self.good_indices_0),
                ] if not one_side else []), dim=1)

            print(f'TOTAL EDGES: {self.neighbors.size(1)}')

            # #Static edge attributes -- dZ, dY, r=sqrt(dZ**2 + dY**2), dFace
            # #TODO  Do things like dWire0, dWire1 make sense for things like cross-pair (i.e. 0,1 and 0,2) neighbors?
            # #      Same question for cross face (i.e. 0,1 on face 0 and 0,1 on face 1)
            

            self.nstatic_edge_attr = 13
            static_edges_0 = self.make_edge_attr(
                nearest_neighbors_0, self.good_indices_0, self.nstatic_edge_attr,
                self.ray_crossings_0, self.nwires_0, 0
            )
            static_edges_1 = self.make_edge_attr(
                nearest_neighbors_1, self.good_indices_1, self.nstatic_edge_attr,
                self.ray_crossings_1, self.nwires_1, 0
            ) 

            self.static_edges = torch.cat(
                [
                    static_edges_0,
                ] + ([
                    static_edges_1,
                ] if not one_side else [])
            )

            self.nchans = [476, 476, 292, 292]

            self.save = True

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
        return edge_attrs
    
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
    
    def forward(self, x):
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
                # self.unets[(i if i < 3 else 2)](xs[i]) for i in range(len(xs))
                checkpoint.checkpoint(
                    self.unets[(i if i < 3 else 2)],
                    xs[i]
                ) for i in range(len(xs))
            ]

            print('passed through unets')
            # for x in xs: print(x.shape)

            #Cat to get into global channel number shape
            x = torch.cat(xs, dim=2)
            if self.save:
                torch.save(x, 'post_unet_test.pt')
            # print('Post unet', x.shape)


        

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
            # print(x.shape)
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
                    
            out = torch.zeros(x.shape[0], 1, nticks_orig, nchannels).to(x.device)
            # roundabout = torch.zeros(x.shape[0], nfeat, nticks_orig, nchannels).to(x.device)
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
                    ncells,
                    nfeat,
                ).to(the_device)


                cross_start = 0
                cross_end = 0
                window_infos = [
                    [as_wires_f0_p0, as_wires_f0_p1, as_wires_f0_p2, self.good_indices_0, self.ray_crossings_0],
                ] + ([] if self.one_side else [
                    [as_wires_f1_p0, as_wires_f1_p1, as_wires_f1_p2, self.good_indices_1, self.ray_crossings_1],
                ])
                for info in window_infos:
                    cross_end += len(info[3])
                    fill_window(
                        window[..., cross_start:cross_end, :],
                        these_nfeats,
                        *info
                    )
                    cross_start = cross_end

                window = window.reshape(-1, nfeat)

                # ##Saving roundabout
                # roundabout_y = window.reshape(nbatches, self.time_window, -1, nfeat)[:, int((self.time_window-1)/2), ...]
                # roundabout_y = [
                #     roundabout_y[:, r0:r1, :] for r0, r1 in ranges
                # ]
                # # roundabout_y = [
                # #     roundabout_y[:, :ncross_01, :],
                # #     roundabout_y[:, ncross_01:ncross_01+ncross_12, :],
                # #     roundabout_y[:, ncross_01+ncross_12:ncross_01+ncross_12+ncross_20:, :],
                # # ]
                # roundabout_y = self.scatter_to_chans(roundabout_y, nbatches, nchannels, the_device)
                # roundabout[:, :, tick, :] = roundabout_y.permute(0, 2, 1)
                # ################


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

                y = [
                    y[:, r0:r1, :] for r0, r1 in ranges
                ]

                temp_out = self.scatter_to_chans(y, nbatches, nchannels, the_device)
                

                #batch, feat, channel
                temp_out = self.mlp(temp_out).view(1,1,-1)
                out[:, 0, tick, :] = temp_out
        

        if self.save:
            # torch.save(roundabout, 'roundabout_test.pt')
            torch.save(out, 'out_test.pt')
            print('Saved')
            self.save = False
        return out.permute(0, 1, 3, 2)

