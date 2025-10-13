import numpy as np
import time
import torch
import torch.nn as nn
from wirecell.dnn.models.unet import UNet
from wirecell.raygrid.coordinates import Coordinates
from wirecell.raygrid import crossover, schema_load
import math
class Network(nn.Module):



    def __init__(self, wires_file='protodunevd-wires-larsoft-v3.json.bz2', nfeatures=4, detector=0):
        super().__init__()
        self.nfeatures=nfeatures
        self.unets = nn.ModuleList([
                UNet(n_channels=2, n_classes=nfeatures,
                     batch_norm=True, bilinear=True, padding=True)
                for i in range(3)
        ])
        self.crossover_term = crossover.CrossoverTerm()
        store = schema_load.StoreDB()
        schema_load.load_file(wires_file, store)
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

        self.face_plane_wires_channels = {}
        for i, face in enumerate(self.faces):
            for j in face.planes:
                plane = store.planes[j]
                # face_plane_wires_channels[(i,j)] = torch.zeros(
                #     (len(plane.wires), 2),
                #     dtype=int
                # )
                wire_chans = []
                for wi in plane.wires:
                    wire = store.wires[wi]
                    wire_chans.append([wire.ident, chanmap[wire.channel]])
                self.face_plane_wires_channels[(i,j)] = torch.tensor(wire_chans, dtype=int)
        

        face_to_plane_to_nwires = {
            i:[len(store.planes[p].wires) for j,p in enumerate(self.faces[i].planes)] for i in face_ids
        }
        print(face_to_plane_to_nwires)
        views_face0 = schema_load.views_from_schema(store, 0)
        coords_face0 = Coordinates(views_face0)
        
        views_face1 = schema_load.views_from_schema(store, 1)
        coords_face1 = Coordinates(views_face1)
        
        self.xover_map0 = crossover.build_map(coords_face0, 0, 1, 2, face_to_plane_to_nwires[0])
        self.xover_map1 = crossover.build_map(coords_face1, 0, 1, 2, face_to_plane_to_nwires[1])
        print('Crossover shape', self.xover_map0.shape)
        torch.save(self.xover_map0, 'xover_map.pt')

        self.xover_map = [self.xover_map0, self.xover_map1]
        
        self.nchans = [476, 476, 292, 292]
        self.segmap = nn.Conv2d(self.nfeatures, 1, 1)

    def forward(self, x):
        '''
        Input data is assumed to be of shape (nbatch, nfeatures, nchannels, nticks)
        '''
        input_shape = x.shape
        the_device = x.device
        print(x.shape)
        xs = [
            x[:, :, (0 if i == 0 else sum(self.nchans[:i])):sum(self.nchans[:i+1]), :]
            for i, nc in enumerate(self.nchans)
        ]
        for x in xs: print(x.shape)

        #Pass through the unets
        xs = [
            self.unets[(i if i < 3 else 2)](xs[i]) for i in range(len(xs))
        ]
        print('passed through unets')
        for x in xs: print(x.shape)

        #Cat to get into global channel number shape
        x = torch.cat(xs, dim=2)

        print(x.shape)

        #Get channels last
        x = x.permute((0,1,3,2))
        print(x.shape)

        nbatches = x.shape[0]
        nfeatures_in = x.shape[1]
        nticks = x.shape[2]
        nchans = x.shape[3]
        expanded = [
            indices.to(the_device).view(1,1,1,*indices.shape).expand(nbatches, nfeatures_in, nticks, -1, -1)
            # indices.to(the_device).view(1,1,1,*indices.shape)
            for indices in self.xover_map
        ]
        #Go from the elec channel view to wire segment view
        wire_seg_view = []
        for iface in range(2):
            
            face = self.faces[iface]
            wire_seg_view.append([])

            for iplane in range(3):
                this_wires_channels = self.face_plane_wires_channels[(iface, face.planes[iplane])]
                view_shape = [i for i in x.shape]
                view_shape[-1] =  len(this_wires_channels)
                print(view_shape)
                wire_seg_view[-1].append(torch.zeros(view_shape).to(the_device))
                print(x[..., this_wires_channels[:,1]].shape)
                wire_seg_view[-1][-1][..., this_wires_channels[:,0]] = x[..., this_wires_channels[:,1]]

        indexed = torch.cat([
            torch.cat(
                [wire_seg_view[i][j][..., self.xover_map[i][:, j]].unsqueeze(-1) for j in range(3)],
                dim=-1
            )
            for i in range(2)
        ], dim=-2)
        print(indexed.shape)

        prev_end = 0
        tick_size = 10
        
        for i in range(math.ceil(indexed.size(-3)/tick_size)):
            # print(i, tick_size*i, tick_size*(i+1))
            # print(indexed)
            indexed[..., tick_size*i:tick_size*(i+1), :] = self.crossover_term(indexed[..., tick_size*i:tick_size*(i+1), :])
        print('Crossover passed')
        print(x.shape)
        
        indexed_plane0 = indexed[..., :len(self.xover_map0), :]
        indexed_plane1 = indexed[..., len(self.xover_map0):, :]
        indexed_arr = [indexed_plane0, indexed_plane1]

        #For each crossing, average the contributions into the wire view
        summed_wire_terms = []
        for i, wsv_face in enumerate(wire_seg_view):
            summed_wire_terms.append([])
            indices = self.xover_map[i]
            expanded = indices.to(the_device).view(1,1,1,*indices.shape).expand(nbatches, nfeatures_in, nticks, -1, -1)
            for j, wsv_plane in enumerate(wsv_face):
                summed = torch.zeros_like(wsv_plane).to(the_device)
                
                # print('Scatter add in sum', i, j, expanded[i][..., j].size(), indexed_arr[i][..., j].size())
                # summed.scatter_add_(-1, index=expanded[i][..., j], src=indexed_arr[i][..., j])
                summed.scatter_add_(-1, index=expanded[..., j], src=indexed_arr[i][..., j])
                
                ##These can probably be precomputed
                count = torch.zeros_like(wsv_plane).to(the_device).detach()
                ones = torch.ones_like(indexed_arr[i][..., j]).to(the_device)
                # count.scatter_add_(-1, index=expanded[i][..., j], src=ones)
                count.scatter_add_(-1, index=expanded[..., j], src=ones)

                summed /= count.clamp(min=1)
                summed_wire_terms[-1].append(summed)
                print(summed.shape)
        
        #Now we have to go to channels view
        summed_y = torch.zeros(nbatches, self.nfeatures, nticks, nchans).to(the_device)
        counts_y = torch.zeros_like(summed_y).to(the_device).detach()

        expanded_wire_channels = {
            ij:wire_channels.to(the_device).view(1,1,1,*wire_channels.shape).expand(nbatches, self.nfeatures, nticks, -1, -1)
            for ij, wire_channels in self.face_plane_wires_channels.items()
        }
        for i in range(len(summed_wire_terms)):
            face = self.faces[i]

            for j in range(len(summed_wire_terms[i])):
                this_wires_channels = expanded_wire_channels[(i, face.planes[j])]
                print(this_wires_channels.shape)
                print(summed_wire_terms[i][j].shape)
                # print(summed_wire_terms[i][j][this_wires_channels[..., -2]].shape)
                ones = torch.ones_like(summed_wire_terms[i][j])
                summed_y.scatter_add_(-1, this_wires_channels[..., -1], summed_wire_terms[i][j])
                counts_y.scatter_add_(-1, this_wires_channels[..., -1], ones)

        summed_y /= counts_y.clamp(min=1)
        summed_y = summed_y.permute(0,1,3,2)
        print(summed_y.shape)



        #Get ticks last
        # x = x.permute((0,1,3,2))
        summed_y = torch.sigmoid(self.segmap(summed_y))
        torch.save(summed_y, f'summed_y_{int(time.time()*1000)}.pt')
        return summed_y
        # return torch.sigmoid(x)

