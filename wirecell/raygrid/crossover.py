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

def build_map(coords, i, j, k, nwires):

    #Make the Ni x Nj pairs of indices in the ith and jth planes
    rays_i = torch.arange(nwires[plane_i])
    rays_j = torch.arange(nwires[plane_j])
    cross = torch.zeros((rays_i.size(0), rays_j.size(0), 2), dtype=int)
    cross[..., 1] = rays_j
    cross = cross.permute(1,0,2)
    cross[..., 0] = rays_i
    cross = cross.reshape((-1, 2))

    #"Real" raygrid views start at index 2
    view1 = plane_i + 2
    view2 = plane_j + 2
    view3 = plane_k + 2

    #Find the locations in the third plane of crossing points from the first 2 planes
    #Then turn those into indices within the last plane
    locs = coords.pitch_location(view1, cross[:, 0], view2, cross[:, 1], view3)
    indices = coords.pitch_index(locs, view3)
    
    #Then check that they are valid a.k.a. they are bounded within the last wire plane
    good_ones = torch.where((indices >= 0) & (indices < nwires[k]))

    #Concat with the corresponding pairs from above
    results = torch.cat((cross[good_ones], indices[good_ones].reshape((-1, 1))), dim=1)
    return results

class CrossoverTerm(nn.Module):
    def __init__(self, map=None, feats=[16,128,]):
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
