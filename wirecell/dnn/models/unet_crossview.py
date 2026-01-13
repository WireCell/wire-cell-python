import numpy as np
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch_scatter
# from torch_geometric.data import Data
# from torch_geometric.nn import GAT, GCN
from wirecell.dnn.models.unet import UNet
from wirecell.dnn.models.scatter import TripleScatterModule
from wirecell.raygrid.coordinates import Coordinates
from wirecell.raygrid import crossover as xover
from wirecell.util.wires import schema, persist
import torch.utils.checkpoint as checkpoint

import math

class UNetCrossView(nn.Module):
    
    def determine_mlp_n_in(self):
        if self.do_call_first_unets:
            mlp_n_in = 3*self.first_unet_n_out
        else:
            mlp_n_in = 3*self.n_input_features
        return mlp_n_in

    def determine_unets2_n_in(self):
        if self.do_call_mp and self.special_style in ['mlp', 'new_scatter']:
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
            if self.do_call_first_unets:
                return 4*self.first_unet_n_out
            else:
                return self.n_input_features

    # @torch.autocast('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16) #TODO -- Fix this
    def __init__(
            self,
 
            wires_file='protodunevd-wires-larsoft-v3.json.bz2',
            chanmap_file='chanmap_1536.npy',
            nchans=[476, 476, 292, 292],
            det_type='vd',
            cells_file=None,
 
            # wires_file='protodunehd-wires-larsoft-v1.json.bz2',
            # chanmap_file=2560,
            # nchans=[800, 800, 480, 480],
            # det_type='hd',
            # cells_file='pdhd_cells.pt',

            mp_out=False,
            scatter_out=False,
            output_as_tuple=False,

            n_unet_features=4,
            checkpoint=True,
            n_feat_wire = 0,
            detector=0,
            n_input_features=1,

            network_style='U',
        ):
        super().__init__()
        self.nfeat_post_unet=n_unet_features
        self.n_feat_wire = n_feat_wire
        self.checkpoint=checkpoint
        self.n_input_features=n_input_features
        
        self.do_call_first_unets = True
        self.do_call_second_unets = True
        self.do_call_mp = True
        self.split_gpu = False
        
        self.split_xview_loop = True
        self.split_at = 17.0

        self.do_call_special = True
        self.split_second_unets = True
        self.special_style = 'new_scatter' #mlp, threshold, or feedthrough
        self.good_specials = ['mlp', 'threshold', 'feedthrough', 'new_scatter']
        if self.special_style not in self.good_specials:
            raise Exception(f'Unknown Special Style {self.special_style}. Can only select one of {(", ").join(self.good_specials)}')
        self.mlp_n_out = 4 #Only used if special_style set to mlp
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.xview_activation = self.relu

        self.scatter_out = scatter_out
        self.mp_out = mp_out
        self.output_as_tuple = output_as_tuple

        self.network_style = network_style
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
        

        self.new_scatter = TripleScatterModule(
            int(mlp_n_in/3), 16, self.mlp_n_out,
            chunk_size=16)

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
            if self.split_second_unets:
                self.unets2 = nn.ModuleList([
                        UNet(n_channels=self.second_unet_n_in, n_classes=1,
                            batch_norm=True, bilinear=True, padding=True)
                        for i in range(3)
                ])
            else:
                self.unets2 = nn.ModuleList([UNet(n_channels=self.second_unet_n_in, n_classes=1,
                                   batch_norm=True, bilinear=True, padding=True)])

            unets2_size = 0
            for m in self.unets2:
                for param in m.parameters():
                    unets2_size += param.nelement() * param.element_size()
                for buffer in m.buffers():
                    unets2_size += buffer.nelement() * buffer.element_size()
            print('TOTAL UNETS2 SIZE', unets2_size)

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
                if i == 0: self.nplanes = len(face.planes)

                for jj, j in enumerate(face.planes):
                    plane = store.planes[j]
                    wire_chans = torch.zeros((len(plane.wires), 2), dtype=int)
                    for wi in plane.wires:
                        wire = store.wires[wi]
                        wire_chans[wire.ident, 0] = wire.ident
                        wire_chans[wire.ident, 1] = chanmap[wire.channel]
                    # self.face_plane_wires_channels[(i,jj)] = torch.tensor(wire_chans, dtype=torch.int)
                    self.face_plane_wires_channels[i*self.nplanes + jj] = torch.tensor(wire_chans, dtype=torch.int)
                    # print("FPWC size", self.face_plane_wires_channels[(i,jj)].dtype)
                    print("FPWC size", self.face_plane_wires_channels[i*self.nplanes + jj].dtype)


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

                print('Cells face 0', self.good_indices_0.shape)
                print('Cells face 1', self.good_indices_1.shape)
            print(self.good_indices_0.dtype, self.good_indices_1.dtype)
            self.good_indices_0 = self.good_indices_0.to(torch.int)
            self.good_indices_1 = self.good_indices_1.to(torch.int)
            print(self.good_indices_0.dtype, self.good_indices_1.dtype)
            self.nchans = nchans #[476, 476, 292, 292]

            self.save = True
    

    def scatter_to_chans(self, y, nbatches, nchannels, the_device):
        #TODO -- check size of y etc
        temp_out = torch.zeros(nbatches, y[0].shape[1], nchannels, y[0].size(-1)).to(the_device).to(y[0].dtype)

        # for k, v in self.face_plane_wires_channels.items():
        #     self.face_plane_wires_channels[k] = v.to(the_device)
        # print('scatter device', the_device)
        # print('Inds devices', self.good_indices_0.device, self.good_indices_1.device)
        # print('input devices', y[0].device, y[1].device)
        # temp_out2 = temp_out.clone()
#         # mem0 = torch.cuda.memory_allocated(0) / (1024**2)
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
#         # mem1 = torch.cuda.memory_allocated(0) / (1024**2)
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
            # temp_out = temp_out.scatter_reduce(
            # print(yi.shape)
            # wcs = wire_chans.to(the_device)[sub_indices.to(the_device)][:,1].unsqueeze(-1).unsqueeze(0).unsqueeze(1).repeat(yi.shape[0], yi.shape[1], 1, yi.shape[-1])
            # print(wcs.nelement()*wcs.element_size())
            # if torch.cuda.is_available():
#             #     allocated_memory = torch.cuda.memory_allocated() / (1024**3)  # in GB
            #     reserved_memory = torch.cuda.memory_reserved() / (1024**3)  # in GB
            #     max_reserved_memory = torch.cuda.max_memory_reserved() / (1024**3) # in GB

            #     print(f"Allocated CUDA memory: {allocated_memory:.2f} GB")
            #     print(f"Reserved CUDA memory: {reserved_memory:.2f} GB")
            #     print(f"Max reserved CUDA memory: {max_reserved_memory:.2f} GB")
            wcs = wire_chans[:,1][indices].unsqueeze(-1).unsqueeze(0).unsqueeze(1).repeat(yi.shape[0], yi.shape[1], 1, yi.shape[-1])
            temp_out = temp_out.scatter_reduce(
                2,
                # wire_chans[indices][:,1].unsqueeze(1).unsqueeze(0).repeat(1, 1, yi.shape[-1]),
                # wire_chans.to(the_device)[sub_indices.to(the_device)][:,1].unsqueeze(-1).unsqueeze(0).unsqueeze(1).repeat(yi.shape[0], yi.shape[1], 1, yi.shape[-1]),
                wcs,
                yi,
                'amax',
                include_self=True,
            )
#             print('Scattering:', torch.cuda.memory_allocated(0) / (1024**2))
            # print(wire_chans.shape, indices.shape)
            
            # torch_scatter.scatter_max(
            #     yi,
            #     # wire_chans.to(the_device)[indices.to(the_device)][:,1],
            #     wire_chans[:,1][indices],
            #     dim=2,
            #     out=temp_out,
            # )
#             print('Scattered:', torch.cuda.memory_allocated(0) / (1024**2))

        # print(temp_out - temp_out2)
        return temp_out

    def make_wires(self, x, face, plane):
        wire_chans = self.face_plane_wires_channels[(face, plane)].to(x.device)
        # print(wire_chans)
        # print('Input', x.shape[0], len(wire_chans), x.shape[-1])
        wires = torch.zeros((x.shape[0], x.shape[1], len(wire_chans), x.shape[-1])).to(x.device).to(x.dtype)
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

    def call_unets(self, x, unets, no_split=False):
        x = self.split_x(x)
        if no_split:
            x = [
                (unets[0](xi) if not self.checkpoint else
                 checkpoint.checkpoint(unets[0], xi)
                ) for xi in x
            ]
        else:
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
        xi = x[0].permute(0, 2, 1)
        print('CALLING NEW SCATTER on', x.shape)
        xi = self.new_scatter(
            xi,
            # [self.face_plane_wires_channels[0,0], self.face_plane_wires_channels[1,0]],
            # [self.face_plane_wires_channels[0,1], self.face_plane_wires_channels[1,1]],
            # [self.face_plane_wires_channels[0,2], self.face_plane_wires_channels[1,2]],
            [self.face_plane_wires_channels[0], self.face_plane_wires_channels[self.nplanes + 0]],
            [self.face_plane_wires_channels[1], self.face_plane_wires_channels[self.nplanes + 1]],
            [self.face_plane_wires_channels[2], self.face_plane_wires_channels[self.nplanes + 2]],
            [self.good_indices_0.T, self.good_indices_1.T]
        ).unsqueeze(0)
        # x = self.crossview_loop(x, call_special=self.do_call_special)
        # x = self.calculate_crossview(x, call_special=self.do_call_special)
        print('DONE', xi.shape)
        xi = self.xview_activation(xi)
        return torch.cat([x, xi.permute(0,1,3,2)], dim=1)

    def A(self, x):
        '''
        Input data is assumed to be of shape (nbatch, nfeatures, nchannels, nticks)
        '''
        input_shape = x.shape
        nbatches = x.shape[0]
        nticks = x.shape[-1]
        nchannels = x.shape[-2]
        print('INPUT DTYPE', x.dtype)
        the_device = x.device
        self.good_indices_0 = self.good_indices_0.to(the_device)
        self.good_indices_1 = self.good_indices_1.to(the_device)
        for k, v in self.face_plane_wires_channels.items():
            self.face_plane_wires_channels[k] = v.to(the_device)
        if self.do_call_second_unets:
            if the_device != 'cpu':
                split_device = self.unet2_device
            else: split_device = 'cpu'
            self.unets2 = nn.ModuleList([ui.to(self.unet2_device) for ui in self.unets2])

        if self.do_call_first_unets:
            print('Calling first UNets')
            # print(torch.cuda.memory_allocated(0) / (1024**2))
            x = self.call_unets(x, self.unets)
            # print(torch.cuda.memory_allocated(0) / (1024**2))

        if self.do_call_mp:
            print('Calling MP')
            # print(torch.cuda.memory_allocated(0) / (1024**2))
            x = self.mp_step(x)
            # print(torch.cuda.memory_allocated(0) / (1024**2))
        if self.do_call_second_unets:
            x = x.to(split_device)
            print('Calling second UNets')
            # print(torch.cuda.memory_allocated(0) / (1024**2))
            x = self.call_unets(x, self.unets2, no_split=(not self.split_second_unets))
            # print(torch.cuda.memory_allocated(0) / (1024**2))
        # x = self.sigmoid(x.to(torch.float32))
        x = self.sigmoid(x)
        xmeta = dict(
            device=the_device,
            nregions=x.shape[-1],
        )
        return x, xmeta

    def B(self, x, xmeta, tick):
        xi = x[..., tick].unsqueeze(-1)
        
        if self.mp_out:
            if self.scatter_out:
                xview = self.calculate_crossview(xi).to(x.device)
            else:
                xview_0, xview_1 = self.make_all_wires(xi, do_cat=False)
                xview = torch.cat([
                    xview_0.to(x.device),
                    xview_1.to(x.device),
                ], dim=-2)

            return (xi, xview)
        else:
            return (xi,) if self.output_as_tuple else xi
    

    def xview_wires_loop(self, x, do_cat=True):
        xview = torch.zeros(x.shape[0], 3*x.shape[1], (len(self.good_indices_0) + len(self.good_indices_1)), x.shape[-1]).to(x.device)
        for i in range(x.shape[-1]):
            # print(i)
            xi = x[..., i].unsqueeze(-1)
            xv = torch.cat(self.make_all_wires(xi, do_cat=do_cat), dim=-2)
            # print(xv_0.shape, xv_1.shape)
            # xview.append(
            #     torch.cat([xv_0, xv_1], dim=-2)
            # )
            # print(xview.shape, xview[..., i].shape)
            xview[..., i] = xv[..., 0]
            # xview[..., i] = torch.cat([xv_0, xv_1], dim=-2)
        # xview = torch.cat(xview, dim=-1)
        return xview

    def forward(self, x):
        x, xmeta = self.A(x)
        if self.mp_out:
            if self.scatter_out:
                x = self.crossview_loop(x).to(xmeta['device'])
                return (x[:, 0].unsqueeze(0), x[:, 1:])
            else:
                xview = self.xview_wires_loop(x, do_cat=False)
                # print(xview.element_size() * xview.nelement())
                return x, xview
        else:
            return (x,) if self.output_as_tuple else x

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
    
    def crossview_loop(self, input, call_special=False):
        crossview_chans = []
        for i in range(input.shape[-1]):
            # print(i)
            xi = input[..., i].unsqueeze(-1)
            mem0 = torch.cuda.memory_allocated(0) / (1024**3)
            mem1 = torch.cuda.memory_allocated(1) / (1024**3)
            # print(f'{mem0:.2f}, {mem1:.2f}')

            crossview_chans.append(
                self.calculate_crossview(xi, call_special=call_special).to(input.device)
            )
            # print(torch.cuda.memory_allocated(0) / (1024**3))
            # print(torch.cuda.memory_allocated(1) / (1024**3))
            # crossview_chans.append(cross_view_all)
        
        x = torch.cat([
            input,
            torch.cat(crossview_chans, dim=-1)
        ], dim=1)
        return x

    def make_all_wires(self, xi, do_cat=True):
        the_device = xi.device
        # crossview_chans = []
        self.good_indices_0 = self.good_indices_0.to(the_device)
        self.good_indices_1 = self.good_indices_1.to(the_device)
        for k, v in self.face_plane_wires_channels.items():
            self.face_plane_wires_channels[k] = v.to(the_device)
        # for i in range(input.shape[-1]):
            # xi = input[..., i].unsqueeze(-1)
            # print(xi.shape)

            #These are called 'as_wires' but are really cells on each face
        if do_cat:
            as_wires_f0 = torch.zeros((xi.shape[0], 3*xi.shape[1], self.good_indices_0.shape[0], xi.shape[-1])).to(the_device)
            # as_wires_f0[:, 0] = self.make_wires(xi, 0, 0)[:, :, self.good_indices_0[:, 0]]
            as_wires_f0[:, 0] = self.make_wires(xi, 0, 0)[:, :, self.good_indices_0[:, 0]]
            as_wires_f0[:, 1] = self.make_wires(xi, 0, 1)[:, :, self.good_indices_0[:, 1]]
            as_wires_f0[:, 2] = self.make_wires(xi, 0, 2)[:, :, self.good_indices_0[:, 2]]

            as_wires_f1 = torch.zeros((xi.shape[0], 3*xi.shape[1], self.good_indices_1.shape[0], xi.shape[-1])).to(the_device)
            as_wires_f1[:, 0] = self.make_wires(xi, 1, 0)[:, :, self.good_indices_1[:, 0]]
            as_wires_f1[:, 1] = self.make_wires(xi, 1, 1)[:, :, self.good_indices_1[:, 1]]
            as_wires_f1[:, 2] = self.make_wires(xi, 1, 2)[:, :, self.good_indices_1[:, 2]]
        else:
            as_wires_f0 = self.make_wires(xi, 0, 0)[:, :, self.good_indices_0[:, 0]]
            as_wires_f0 *= self.make_wires(xi, 0, 1)[:, :, self.good_indices_0[:, 1]]
            as_wires_f0 *= self.make_wires(xi, 0, 2)[:, :, self.good_indices_0[:, 2]]

            as_wires_f1 =  self.make_wires(xi, 1, 0)[:, :, self.good_indices_1[:, 0]]
            as_wires_f1 *= self.make_wires(xi, 1, 1)[:, :, self.good_indices_1[:, 1]]
            as_wires_f1 *= self.make_wires(xi, 1, 2)[:, :, self.good_indices_1[:, 2]]
        return as_wires_f0, as_wires_f1

    def calculate_crossview(self, xi, call_special=False):
        #Now we have to construct MP3 and MP2
        #Go from the values on the channels to wires then make cells
        ##HAVE TO DO THIS IN A LOOP BECAUSE IT'S TOO BIG
        ## SO INSTEAD OF DOING EVERY TIME SAMPLE AT ONCE, DO IT IN A LOOP FILL THE TIME SAMPLE OUTPUT
        ## ONE BY ONE
        # input = input.to('cuda:1')
        mem0 = torch.cuda.memory_allocated(0) / (1024**3)
        mem1 = torch.cuda.memory_allocated(1) / (1024**3)
        the_device = xi.device
        if self.split_xview_loop and self.training:
            the_device = 'cuda:0' if (mem0 < self.split_at) else 'cuda:1'
            xi = xi.to(the_device)
        nbatches = xi.shape[0]
        nchannels = xi.shape[-2]
        # the_device = xi.device
        # crossview_chans = []
        self.good_indices_0 = self.good_indices_0.to(the_device)
        self.good_indices_1 = self.good_indices_1.to(the_device)
        for k, v in self.face_plane_wires_channels.items():
            self.face_plane_wires_channels[k] = v.to(the_device)

        as_wires_f0, as_wires_f1 = self.make_all_wires(xi)

        #This mixes the information between the three planes
        if call_special:
            if self.special_style == 'mlp':
                print(the_device, as_wires_f0.device, as_wires_f1.device)
                as_wires_f0 = self.call_mlp(as_wires_f0.to(next(self.mlp.parameters()).device)).to(the_device)
                as_wires_f1 = self.call_mlp(as_wires_f1.to(next(self.mlp.parameters()).device)).to(the_device)
            elif self.special_style == 'threshold':
                # print('Calling threshold')
                as_wires_f0 = self.call_threshold_like(as_wires_f0)
                as_wires_f1 = self.call_threshold_like(as_wires_f1)

        cross_view_all = self.scatter_to_chans(
            [as_wires_f0, as_wires_f1],
            nbatches, nchannels, the_device
        )
        return cross_view_all
        # crossview_chans.append(cross_view_all)
        
        # x = torch.cat([
        #     input,
        #     torch.cat(crossview_chans, dim=-1)
        # ], dim=1)
        # return x #.to('cuda:0')

    def make_label_nodes(self, labels):
        with torch.no_grad():
            return self.calculate_crossview(labels)

    def make_label_nodes_full(self, labels):
        '''
        We get the labels in a shape of (batch, feat=1, channels, ticks)

        So we need to go from channels view to the wires then nodes view.
        '''
        with torch.no_grad():
            if self.mp_out:
                if self.scatter_out:
                    labels = self.crossview_loop(labels)
                    return (labels[:, 0].unsqueeze(0), labels[:, 1:])
                else:
                    label_wires = torch.cat(self.make_all_wires(labels, do_cat=False), dim=-2)
                    # print('LW shapes', label_wires_0.shape, label_wires_1.shape)
                    return labels, label_wires
            else:
                return (labels,) if self.output_as_tuple else labels
