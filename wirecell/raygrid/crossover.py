import sys
import torch
from torch import nn
from math import sqrt
from wirecell.raygrid import (
    plots
)
from wirecell.raygrid.coordinates import Coordinates
from wirecell.raygrid.examples import (
    symmetric_views, random_points, random_groups, fill_activity
)
import matplotlib.pyplot as plt 
import json 
from argparse import ArgumentParser as ap

def scatter_crossings(coords, v1, r1, v2, r2):
    rc = coords.ray_crossing(v1, r1, v2, r2)
    xs = rc.detach().numpy()[:,0]
    ys = rc.detach().numpy()[:,1]
    plt.scatter(xs, ys, color='black')

def coords_from_schema(store, face_index, drift='vd'):
    views = views_from_schema(store, face_index, drift)
    coords = Coordinates(views)
    return coords

def get_center(store, wire, drift='vd'):

    head = store.points[wire.head]
    tail = store.points[wire.tail]
    center =  torch.mean(torch.Tensor([
            [head.z, head.y],
            [tail.z, tail.y],
    ]), dim=0)
    return center

def draw_schema(store, face_index, plane_indices=[0,1,2], colors=['orange','blue','red']):
    import matplotlib.pyplot as plt
    import numpy as np
    
    planes = []
    for pi in plane_indices:
        global_plane = store.faces[face_index].planes[pi]
        plane = store.planes[global_plane]
        planes.append(plane)
        for wi in plane.wires:
            wire = store.wires[wi]
            head = store.points[wire.head]
            tail = store.points[wire.tail]
            xs = [tail.z, head.z]
            ys = [tail.y, head.y]
            plt.plot(xs, ys, color=colors[pi])
    plt.show()

def views_from_schema(store, face_index, drift='vd'):
    #GEt the plane objects from the store for htis face
    planes = [store.planes[i] for i in store.faces[face_index].planes]


    views = []
    #For each plane, get the first and second wire to get the pitch direction and magnitude
    for plane in planes:
        first_wire = store.wires[plane.wires[0]]
        second_wire = store.wires[plane.wires[1]]
        # print(first_wire, second_wire)

        first_center = get_center(store, first_wire)
        # print(first_center)

        second_center = get_center(store, second_wire)
        # print(second_center)
        views.append(torch.cat([first_center.unsqueeze(0), second_center.unsqueeze(0)], dim=0).unsqueeze(0))

    return torch.cat(views)

def build_cross(rays_i, rays_j):
    cross = torch.zeros((rays_i.size(0), rays_j.size(0), 2), dtype=int)
    cross[..., 1] = rays_j
    cross = cross.permute(1,0,2)
    cross[..., 0] = rays_i
    cross = cross.reshape((-1, 2))
    return cross

def get_indices(coords, cross, i, j, k):
    #Find the locations in the third plane of crossing points from the first 2 planes
    #Then turn those into indices within the last plane
    locs = coords.pitch_location(i, cross[:, 0], j, cross[:, 1], k)
    # torch.save(locs, 'xover_locs.pt')
    indices = coords.pitch_index(locs, k)
    return indices


def build_map(coords, plane_i, plane_j, plane_k, nwires):

    #Make the Ni x Nj pairs of indices in the ith and jth planes
    rays_0 = torch.arange(nwires[0])
    rays_1 = torch.arange(nwires[1])
    rays_2 = torch.arange(nwires[2])
    
    rays = [rays_0, rays_1, rays_2]

    # cross = torch.zeros((rays_i.size(0), rays_j.size(0), 2), dtype=int)
    # cross[..., 1] = rays_j
    # cross = cross.permute(1,0,2)
    # cross[..., 0] = rays_i
    # cross = cross.reshape((-1, 2))
    # cross = build_cross(rays_0, rays_1)

    cross01 = build_cross(rays_0, rays_1)
    cross12 = build_cross(rays_1, rays_2)
    cross20 = build_cross(rays_2, rays_0)
    
    # print(cross.shape, 'Crossings')
    # torch.save(cross, 'xover_cross.pt')
    #"Real" raygrid views start at index 2
    view0 = 0 + (len(coords.views) - 3)
    view1 = 1 + (len(coords.views) - 3)
    view2 = 2 + (len(coords.views) - 3)

    #Find the locations in the third plane of crossing points from the first 2 planes
    #Then turn those into indices within the last plane
    # locs = coords.pitch_location(view0, cross[:, 0], view1, cross[:, 1], view2)
    # torch.save(locs, 'xover_locs.pt')
    # indices = coords.pitch_index(locs, view2)

    indices012 = get_indices(coords, cross01, view0, view1, view2)
    indices120 = get_indices(coords, cross12, view1, view2, view0)
    indices201 = get_indices(coords, cross20, view2, view0, view1)

    good_ones012 = torch.where((indices012 >= 0) & (indices012 < nwires[2]))
    good_ones120 = torch.where((indices120 >= 0) & (indices120 < nwires[0]))
    good_ones201 = torch.where((indices201 >= 0) & (indices201 < nwires[1]))

    results012 = torch.cat((cross01[good_ones012], indices012[good_ones012].reshape((-1, 1))), dim=1)
    results120 = torch.cat((cross12[good_ones120], indices120[good_ones120].reshape((-1, 1))), dim=1)
    results201 = torch.cat((cross20[good_ones201], indices201[good_ones201].reshape((-1, 1))), dim=1)

    results = torch.zeros((results012.shape[0] + results120.shape[0] + results201.shape[0], 3), dtype=int)
    results[:results012.shape[0],:] = results012

    results[results012.shape[0]:results012.shape[0]+results120.shape[0], 0] = results120[:, 2]
    results[results012.shape[0]:results012.shape[0]+results120.shape[0], 1] = results120[:, 0]
    results[results012.shape[0]:results012.shape[0]+results120.shape[0], 2] = results120[:, 1]

    results[-results201.shape[0]:, 0] = results201[:, 1]
    results[-results201.shape[0]:, 1] = results201[:, 2]
    results[-results201.shape[0]:, 2] = results201[:, 0]


    # torch.save(locs, 'xover_indices.pt')
    #Then check that they are valid a.k.a. they are bounded within the last wire plane
    # good_ones = torch.where((indices >= 0) & (indices < nwires[plane_k]))

    #Concat with the corresponding pairs from above
    # results = torch.cat((cross[good_ones], indices[good_ones].reshape((-1, 1))), dim=1)

    # results = torch.unique(torch.cat([results012, results120, results201], dim=0), dim=0)
    results = torch.unique(results, dim=0)
    return results

class CrossoverTerm(nn.Module):
    def __init__(self, map=None, feats=[16,32,]):
        super().__init__()
        self.map = map
        # 3-in, 3-out
        feats_with_3 = [3] + feats + [3]
        self.linears = [nn.Linear(*(feats_with_3[i:i+2])) for i in range(len(feats)+1)]
        self.nodes = nn.Sequential(*self.linears)

        # Need to go from the mapped 

    # This currently takes in already-mapped images.
    # As in: we've taken the ntick x nwires images on the three planes,
    # and extracted the values for each trio of wire indices.
    #
    # 1) That can be moved into here. 
    # 2) Then also we need to take values from duplicated indices 
    #    and average them out
    #
    def forward(self, x):
        return self.nodes(x)

if __name__ == '__main__':
    parser = ap()
    args = parser.parse_args()

    width = 100.
    height = 100.
    pitch_mag = 3.

    views = symmetric_views(width=width, height=height, pitch_mag=pitch_mag)
    coords = Coordinates(views)
    
    #Get the nwires of each view -- Need to check if this is correct
    #Symmetric views starts with / wires, then \, then |
    # / wires start at upper left
    dx = width - views[2][0][0].item()
    dy = -views[2][0][1].item()
    nwires_0 = int(sqrt(dx**2 + dy**2)/pitch_mag) + 1
    nwires_1 = nwires_0
    nwires_2 = int(width/pitch_mag)
    nwires = [nwires_0, nwires_1, nwires_2]

    #Make random images
    nticks = 100
    nbatches = 4
    nfeatures = 3
    img = [
        torch.randn((nbatches, nfeatures, nticks, nw))
        for i, nw in enumerate(nwires)
    ]

    #TODO -- add line to img
    # line = [[width/4, height/2], [3*width/4, height/2]]



    for i in range(3):
        plt.subplot(2,3,1+i)
        plt.imshow(img[i][0, 0], aspect='auto')

    print('nwires:', nwires_0, nwires_1, nwires_2)
    print(views)
    # print(coords)
    plane_i = 0
    plane_j = 1
    plane_k = 2
    indices = build_map(coords, plane_i, plane_j, plane_k, [nwires_0, nwires_1, nwires_2])
    print(indices, indices.nelement(), indices.nelement()*indices.element_size())

    # for ii, p in enumerate([plane_i, plane_j, plane_k]):
    #     print(ii, img[ii][:, indices[:,p]].shape)

    #For each image, and within each image, for each batch x feature x tick,
    # map out the values
    indexed = [
        img[ii][..., indices[:,p]].unsqueeze(-1)
        for ii, p in enumerate([plane_i, plane_j, plane_k])
    ]
    # Then concatenate them together
    catted = torch.cat(indexed, dim=-1)

    for i in indexed: print(i.shape)

    xover_term = CrossoverTerm(indices)
    print('Passing')
    passed = xover_term(catted)
    print(passed.shape)
    print(passed.element_size()*passed.nelement()*1e-6)

    expanded = indices.view(1,1,1,*indices.shape).expand(nbatches, nfeatures, nticks, -1, -1)
    print(expanded.shape)

    output = [torch.zeros_like(i) for i in img]


    for ii, p in enumerate([plane_i, plane_j, plane_k]):
        sum = torch.zeros_like(img[ii])
        sum.scatter_add_(-1, expanded[..., p], passed[...,ii])
    
        count = torch.zeros_like(img[ii])
        ones = torch.ones_like(passed[..., ii])
        count.scatter_add_(-1, expanded[..., p], ones)

        sum /= count.clamp(min=1)


        print(sum.shape)
        plt.subplot(2,3, 1+ii + 3)
        plt.imshow(sum[0, 0].detach().numpy(), aspect='auto')
    

    plt.show()
