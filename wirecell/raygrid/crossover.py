import torch
from wirecell.raygrid.coordinates import Coordinates
from wirecell.raygrid.examples import symmetric_views
import json 
from argparse import ArgumentParser as ap

def build_map(coords, i, j, k):

    #Get the number of wires in each view
    #  These will be bounded by the first two views

    # for testing
    nwires = [100,100,100]

    rays_i = torch.arange(nwires[plane_i])
    rays_j = torch.arange(nwires[plane_j])
    cross = torch.zeros((rays_i.size(0), rays_j.size(0), 2), dtype=int)
    cross[..., 1] = rays_j
    cross = cross.permute(1,0,2)
    cross[..., 0] = rays_i
    cross = cross.reshape((-1, 2))

    view1 = plane_i + 2
    view2 = plane_j + 2
    view3 = plane_k + 2

    locs = coords.pitch_location(view1, cross[:, 0], view2, cross[:, 1], view3)
    indices = coords.pitch_index(locs, view3)

    good_ones = torch.where((indices >= 0) & (indices < nwires[k]))
    results = torch.cat((cross[good_ones], indices[good_ones].reshape((-1, 1))), dim=1)
    return results

if __name__ == '__main__':
    parser = ap()
    args = parser.parse_args()

    views = symmetric_views()
    coords = Coordinates(views)
    print(views)
    print(coords)
    plane_i = 0
    plane_j = 2
    plane_k = 1
    indices = build_map(coords, plane_i, plane_j, plane_k)
    print(indices.shape)
    
