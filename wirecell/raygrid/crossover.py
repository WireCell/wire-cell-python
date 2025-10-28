import sys
import torch
from torch import nn
from math import sqrt
from wirecell.raygrid import (
    plots
)
from wirecell.raygrid.coordinates import Coordinates
import wirecell.raygrid.tiling as tiling
from wirecell.raygrid.examples import (
    symmetric_views, random_points, random_groups, fill_activity
)
import matplotlib.pyplot as plt 
import json 
from argparse import ArgumentParser as ap
import numpy as np

def apply_sequence(coords, nw, blobs_in, run=1, warn=False):
    blobs_out = []
    wires = torch.zeros(nw).to(bool)
    for i in range(0, nw, run):
        wires[i:i+run] = 1
        blobs_i = tiling.apply_activity(coords, blobs_in, wires)
        if blobs_i.size(0) > 0:
            blobs_out.append(blobs_i)
        elif warn:
            print('WARNING SIZE 0 FROM BLOB')
        wires[i:i+run] = 0
    return torch.cat(blobs_out)

def make_cells(coords, nw_0, nw_1, nw_2):
    trivial_blobs = tiling.trivial_blobs()

    all_cells = []
    wires_0 = torch.zeros(nw_0).to(bool)
    wires_1 = torch.zeros(nw_1).to(bool)
    wires_2 = torch.zeros(nw_2).to(bool)

    blobs = apply_sequence(coords, nw_0, trivial_blobs, warn=True)
    blobs = apply_sequence(coords, nw_1, blobs)
    blobs = apply_sequence(coords, nw_2, blobs)
    return blobs[:, 2:, 0]

def get_nearest(rcs, n=5):
    dist = (rcs[:, 0].unsqueeze(0) - rcs[:,0].unsqueeze(1))**2
    dist += (rcs[:, 1].unsqueeze(0) - rcs[:,1].unsqueeze(1))**2
    dist += 1.e32*torch.eye(len(rcs))

    knns = torch.zeros((n, len(rcs)))
    #nearest_dists = []
    arange = torch.arange(len(rcs))
    for i in range(n):
        nearest = dist.argmin(dim=1)
        # , dist[(arange, nearest)] -- nearest distances
        # knns[0, :] = arange
        knns[i, :] = nearest
        # knns.append((arange, nearest))
        dist[(arange, nearest)] += 1.e32
    return knns

def scatter_crossings(coords, v1, r1, v2, r2, color='black'):
    rc = coords.ray_crossing(v1, r1, v2, r2)
    xs = rc.detach().numpy()[:,0]
    ys = rc.detach().numpy()[:,1]
    plt.scatter(xs, ys, color=color)

def coords_from_schema(store, face_index, drift='vd'):
    views = views_from_schema(store, face_index, drift)
    coords = Coordinates(views.to(torch.float))
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
    min_y, max_y = sys.float_info.max, -sys.float_info.max
    min_z, max_z = sys.float_info.max, -sys.float_info.max
    for plane in planes:
        first_wire = store.wires[plane.wires[0]]
        second_wire = store.wires[plane.wires[1]]
        # print(first_wire, second_wire)

        first_center = get_center(store, first_wire)
        second_center = get_center(store, second_wire)
        
        first_head = store.points[first_wire.head]
        first_tail = store.points[first_wire.tail]

        second_head = store.points[second_wire.head]
        second_tail = store.points[second_wire.tail]

        first_head = np.array([first_head.x, first_head.y, first_head.z])
        first_tail = np.array([first_tail.x, first_tail.y, first_tail.z])
        second_head = np.array([second_head.x, second_head.y, second_head.z])
        second_tail = np.array([second_tail.x, second_tail.y, second_tail.z])

        for wi in plane.wires:
            wire = store.wires[wi]
            head = store.points[wire.head]
            tail = store.points[wire.tail]

            min_y = min([head.y, tail.y, min_y])
            max_y = max([head.y, tail.y, max_y])
            min_z = min([head.z, tail.z, min_z])
            max_z = max([head.z, tail.z, max_z])

        b = first_head - first_tail
        b = b / np.linalg.norm(b)

        pitch = np.linalg.norm(np.linalg.cross((second_head - first_head), b))
        print(pitch)

        b = pitch * np.array([b[2], b[1]])
        b = np.linalg.matmul(
            b, [[0, -1], [1, 0]]
        )

        second_point = first_center + b
             
        # print(first_center)

        # print(second_center)
        # views.append(torch.cat([first_center.unsqueeze(0), second_center.unsqueeze(0)], dim=0).unsqueeze(0))
        views.append(torch.cat([first_center.unsqueeze(0), second_point.unsqueeze(0)], dim=0).unsqueeze(0))

    ul = torch.Tensor([min_z, max_y])
    ur = torch.Tensor([max_z, max_y])
    ll = torch.Tensor([min_z, min_y])
    lr = torch.Tensor([max_z, min_y])
    view_0 = torch.cat([
        ((ur + ul)/2).unsqueeze(0), 
        ((ll + lr)/2).unsqueeze(0), 
    ], dim=0).unsqueeze(0)
    view_1 = torch.cat([
        ((ul + ll)/2).unsqueeze(0), 
        ((ur + lr)/2).unsqueeze(0), 
    ], dim=0).unsqueeze(0)
    views = [view_0, view_1] + views
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
    base = len(coords.views) - 3
    locs = coords.pitch_location(i+base, cross[:, 0], j+base, cross[:, 1], k+base)
    # torch.save(locs, 'xover_locs.pt')
    indices = coords.pitch_index(locs, k+base)
    return indices

def get_good_crossers(coords, i, j, nwires):
    cross = build_cross(torch.arange(nwires[i]), torch.arange(nwires[j]))
    base = len(coords.views) - 3
    ray_crossings = coords.ray_crossing(i+base, cross[:, 0], j+base, cross[:, 1])

    #Check that they are within the bounding box
    good = torch.where(
        (ray_crossings[:,1] >= coords.bounding_box[1,0]) &
        (ray_crossings[:,1] <  coords.bounding_box[1,1]) &
        (ray_crossings[:,0] >= coords.bounding_box[0,0]) &
        (ray_crossings[:,0] <  coords.bounding_box[0,1])
    )

    return cross[good] #, ray_crossings[good]

def build_map(coords, nwires, unique=True):

    trios = [(0,1,2), (1,2,0), (2,0,1)]
    good_crossers = [get_good_crossers(coords, i, j, nwires) for i,j,k in trios]
    good_indices = [
        get_indices(
            coords,
            crossers,
            i, j, k
         ) for crossers, (i,j,k) in zip(good_crossers, trios)
    ]

    in_ranges = [torch.where((gi >= 0) & (gi < nwires[i])) for gi, i in zip(good_indices, [2,0,1])]
    good_crossers = [gc[ir] for gc,ir in zip(good_crossers, in_ranges)]
    good_indices = [gc[ir] for gc,ir in zip(good_indices, in_ranges)]

    results = []
    for i in range(len(trios)):
        results.append(torch.zeros(len(good_indices[i]), 3).to(int))
    #0,1,2
    results[0][:, 0] = good_crossers[0][:,0]
    results[0][:, 1] = good_crossers[0][:,1]
    results[0][:, 2] = good_indices[0]
    
    #1,2,0
    results[1][:, 0] = good_indices[1]
    results[1][:, 1] = good_crossers[1][:,0]
    results[1][:, 2] = good_crossers[1][:,1]

    #2,0,1
    results[2][:, 0] = good_crossers[2][:,1]
    results[2][:, 1] = good_indices[2]
    results[2][:, 2] = good_crossers[2][:,0]

    results = torch.cat(results)
    if unique: results = torch.unique(results, dim=0)
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
