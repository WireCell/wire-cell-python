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

def make_chanmap(chanmap_name, store, face_ids=[0,1]):
    chanmap_npy = np.load(chanmap_name)

    #maps from chanident to index in input arrays
    chanmap = {c:i for i, c in chanmap_npy}
    faces = [store.faces[f] for f in face_ids]
    #Build the map to go between wire segments & channels 
    results = {}
    for i, face in enumerate(faces):
        for jj, j in enumerate(face.planes):
            plane = store.planes[j]
            wire_chans = torch.zeros((len(plane.wires), 2), dtype=int)
            for wi in plane.wires:
                wire = store.wires[wi]
                wire_chans[wire.ident, 0] = wire.ident
                wire_chans[wire.ident, 1] = chanmap[wire.channel]
            results[(i,jj)] = wire_chans
    return results

def get_ordered_inverse_indices(S: torch.Tensor):
    """
    Finds unique elements in S and the order-preserving inverse indices.
    
    The inverse_indices map: S[i] -> ordered_unique_elements[j], 
    where j is based on the order of first appearance in S.
    
    Args:
        S: Tensor of shape [n1, 5, 2].
        
    Returns:
        A tuple: (ordered_unique_elements, ordered_inverse_indices)
    """
    
    # 1. Flatten the elements (5, 2) into (10) for row comparison
    S_flat = S.view(S.shape[0], -1) 
    
    # Use torch.unique to get the unique set (sorted by value) and the inverse map (by value)
    unique_elements_flat_val_sorted, inverse_indices_val_mapped, _ = torch.unique(
        S_flat, 
        dim=0, 
        sorted=True, 
        return_inverse=True, 
        return_counts=True
    )
    
    N_unique = unique_elements_flat_val_sorted.shape[0]
    n1 = S.shape[0]
    
    # 2. Find the FIRST index of appearance for each unique element ID (which is currently value-sorted)
    original_S_indices = torch.arange(n1, device=S.device)
    
    # Use scatter_reduce_ to find the minimum (first) index of appearance for each unique ID
    # This is highly efficient and memory safe.
    first_occurrence_indices = torch.full((N_unique,), n1, dtype=torch.long, device=S.device)
    first_occurrence_indices.scatter_reduce_(
        dim=0, 
        index=inverse_indices_val_mapped, 
        src=original_S_indices, 
        reduce="amin"
    )
    
    # 3. Get the permutation that sorts the unique IDs by their first appearance
    sort_permutation = torch.argsort(first_occurrence_indices, descending=False)
    
    # Apply the permutation to the unique elements
    ordered_unique_elements_flat = unique_elements_flat_val_sorted[sort_permutation]
    ordered_unique_elements = ordered_unique_elements_flat.view(-1, 5, 2)
    
    # 4. Re-map the inverse_indices to the new order
    # remap_indices: maps old_unique_id (value-sorted) -> new_unique_id (order-preserving index)
    remap_indices = torch.argsort(sort_permutation) 
    
    # The final, order-preserving inverse indices
    ordered_inverse_indices = remap_indices[inverse_indices_val_mapped]
    
    return ordered_unique_elements, ordered_inverse_indices

def downsample_blobs(highres_blobs, to_run=2):
    
    '''
    Takes in blobs of some resolution and attempts to downsample to a lower resolution
    (in terms of number of wires in each plane). It's pretty dumb so far, and any progressive
    downsampling works best (at all?) in sequential powers of the same factor.

    i.e. you should do to_run = 2 --> 4 --> 8 --> 16

    This is because blobs won't 'line up' nicely otherwise
    for example I tried doing from 16-->100 once and it made a blob span (0, 200)
    and not (0, 100), (100, 200)
    '''

    results = highres_blobs.clone()
    print(results.shape)

    print(results[0])
    base = results.shape[1] - 3
    print(base)

    for i in range(base, base+3):
        max = torch.max(results[:,i])
        print('Plane', i, max)
        # evens = torch.where(1 - results[:, i, 0] % 2)[0]
        # odds = torch.where(results[:, i, 0] % 2)[0]

        # results[odds, i, 0] -= 1
        # results[evens, i, 1] += 1
        results[:, i, 0] = torch.floor(results[:, i, 0]/to_run)*to_run
        results[:, i, 1] = torch.ceil(results[:, i, 1]/to_run)*to_run
        results[:, i] = torch.clamp(results[:, i], 0, max)
    
    #Returns tuple with unique values + indices of source in unique output
    results = get_ordered_inverse_indices(results)
    return results



def draw_pred_comp(p, t):
    num_categories = 4
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    colors = torch.tensor([
        [0., 0., 0., 1.], #black
        [1., 0., 0., 1.], #red
        [1., 1., 1., 1.], #white
        [0., 0., 1., 1.], #blue
    ])
    custom_cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(num_categories + 1)
    norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)
    c = torch.zeros_like(t[0,0]).to(int) #Default -> missed real pixel
    c[torch.where((t[0,0] == 0) & ((p[0,0]>.5) == 1))] = 1 #Fake pixel
    c[torch.where((t[0,0] == 0) & ((p[0,0]<.5)))] = 2 #No prediction empty
    c[torch.where((t[0,0] == 1) & ((p[0,0]>.5)))] = 3 #Predicted real pixel

    patches = [mpatches.Patch(color=colors[i], label=l) for i, l in enumerate(['Miss', 'Fake', 'Empty', 'Real'])]
    plt.imshow(c, cmap=custom_cmap, norm=norm, aspect='auto', interpolation='none')
    plt.legend(handles=patches)
    plt.show()

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

def make_cells(coords, nw_0, nw_1, nw_2, keep_shape=False):
    trivial_blobs = tiling.trivial_blobs()

    all_cells = []
    wires_0 = torch.zeros(nw_0).to(bool)
    wires_1 = torch.zeros(nw_1).to(bool)
    wires_2 = torch.zeros(nw_2).to(bool)

    blobs = apply_sequence(coords, nw_0, trivial_blobs, warn=True)
    blobs = apply_sequence(coords, nw_1, blobs)
    blobs = apply_sequence(coords, nw_2, blobs)
    return (blobs if keep_shape else blobs[:, 2:, 0])

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

def coords_from_schema(store, face_index, shift_half=True):
    views = views_from_schema(store, face_index, shift_half=shift_half)
    coords = Coordinates(views.to(torch.float64))
    return coords

def get_center(store, wire):

    head = store.points[wire.head]
    tail = store.points[wire.tail]
    center =  torch.mean(torch.Tensor([
            [head.z, head.y],
            [tail.z, tail.y],
    ]), dim=0)
    return center

def draw_schema(store, face_index, plane_indices=[0,1,2], colors=['orange','blue','red'], highlight=dict()):
    import matplotlib.pyplot as plt
    import numpy as np
    
    planes = []
    for pi in plane_indices:
        global_plane = store.faces[face_index].planes[pi]
        plane = store.planes[global_plane]
        planes.append(plane)
        to_highlight = highlight[pi] if pi in highlight else []
        for i, wi in enumerate(plane.wires):
            print(i,wi)
            wire = store.wires[wi]
            head = store.points[wire.head]
            tail = store.points[wire.tail]
            xs = [tail.z, head.z]
            ys = [tail.y, head.y]
            plt.plot(xs, ys, color=colors[pi], linestyle=('dashed' if i in to_highlight else 'solid'))
    plt.show()

def draw_blobs(store, face_index, rays=None, plane_indices=[0,1,2], colors=['orange','blue','red'], alpha=0.3, limits=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np
    from math import pi, sqrt, atan, tan, cos, sin
    coords = coords_from_schema(store, face_index)

    face = store.faces[face_index]
    plane_objs = [store.planes[p] for p in face.planes]
    nwires = [len(p.wires) for p in plane_objs]
    print(nwires)
    bounds = coords.bounding_box

    bl = (bounds[0][0], bounds[1][0])
    width = abs(coords.views[1][1][0] - coords.views[1][0][0])
    height = abs(coords.views[0][1][1] - coords.views[0][0][1])
    print(bl, width, height)
    fig, ax = plt.subplots()
    rect = Rectangle(bl, width, height, facecolor='grey', edgecolor='black', alpha=0.7)
    ax.add_patch(rect)
    ax.set_aspect('equal', adjustable='box') 

    if limits is None:
        ax.set_xlim(bl[0] - 1., bl[0] + width + 1.)
        ax.set_ylim(bl[1] - 1., bl[1] + height + 1.)
    else:
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])

    for plane in range(2, 5):
        pitch_mag = coords.pitch_mag[plane]
        pitch_dir = coords.pitch_dir[plane]
        xy = coords.views[plane][0]
        a=0
        these_rays = rays[plane-2]
        for i in these_rays:
            this_xy = xy + i*pitch_mag*pitch_dir
            print(this_xy, pitch_mag, pitch_dir)
            theta = (180./pi)*atan(coords.pitch_dir[plane][1]/coords.pitch_dir[plane][0]) - 90.
            ax.axline(this_xy, slope=(tan(theta*pi/180.)), color=colors[plane-2])
            # r = Rectangle(
            #     xy=(this_xy[0].item(), this_xy[1].item()-2000.),
            #     width=coords.pitch_mag[plane],
            #     height=2*sqrt(coords.pitch_mag[0]**2 + coords.pitch_mag[1]**2),
            #     angle=(180./pi)*atan(coords.pitch_dir[plane][1]/coords.pitch_dir[plane][0]),
            #     rotation_point=(this_xy[0].item(), this_xy[1].item()),
            #     alpha=alpha,
            #     color=colors[plane-2]
            # )
            # ax.add_patch(r)
            # ax.scatter(xy[0].item(), xy[1].item())
            # xy += pitch_mag*pitch_dir
            # break
        # break
    # for pi in plane_indices:
    #         plt.plot(xs, ys, color=colors[pi])
    plt.show()

def make_poly(coords, plane, ray_low, ray_high):
    from math import pi, sqrt, atan, tan, cos, sin
    from shapely.geometry import Polygon
    base=2
    plane += base
    pitch_mag = coords.pitch_mag[plane]
    pitch_dir = coords.pitch_dir[plane]
    xy = coords.views[plane][0] + ray_low*pitch_dir*pitch_mag
    theta = atan(pitch_dir[1]/pitch_dir[0]) - pi/2
    p0 = (xy + 10000.*torch.Tensor([cos(theta), sin(theta)]))
    p1 = (xy - 10000.*torch.Tensor([cos(theta), sin(theta)]))
    p2 = p0 + (ray_high - ray_low)*pitch_dir*pitch_mag
    p3 = p1 + (ray_high - ray_low)*pitch_dir*pitch_mag
    poly = Polygon([p0, p1, p3, p2])
    return poly

def make_poly_insitu(coords, plane, ray_low, ray_high, extent=1.e5):
    from math import pi, sqrt, atan, tan, cos, sin
    base=2
    plane += base
    pitch_mag = coords.pitch_mag[plane]
    pitch_dir = coords.pitch_dir[plane]
    xy = coords.views[plane][0] + ray_low*pitch_dir*pitch_mag
    theta = atan(pitch_dir[1]/pitch_dir[0]) - pi/2
    p0 = (xy + extent*torch.Tensor([cos(theta), sin(theta)]))
    p1 = (xy - extent*torch.Tensor([cos(theta), sin(theta)]))
    p2 = p1 + (ray_high - ray_low)*pitch_dir*pitch_mag
    p3 = p0 + (ray_high - ray_low)*pitch_dir*pitch_mag

    #NEED TO ORDER THESE CCW
    poly = torch.vstack([p0.unsqueeze(0), p1.unsqueeze(0), p2.unsqueeze(0), p3.unsqueeze(0)])

    return poly

def make_poly_from_blob(coords, blob, has_trivial=False):

    sub = 2 if has_trivial else 0
    polys = [make_poly(coords, i-sub, *blob[i]) for i in range(blob.shape[0])]
    poly = polys[0]
    for p in polys[1:]: poly = poly.intersection(p)
    return poly


def views_from_schema(store, face_index, shift_half=True):
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

        first_head =  torch.tensor([first_head.x, first_head.y, first_head.z])
        first_tail =  torch.tensor([first_tail.x, first_tail.y, first_tail.z])
        second_head = torch.tensor([second_head.x, second_head.y, second_head.z])
        second_tail = torch.tensor([second_tail.x, second_tail.y, second_tail.z])
       

        for wi in plane.wires:
            wire = store.wires[wi]
            head = store.points[wire.head]
            tail = store.points[wire.tail]

            min_y = min([head.y, tail.y, min_y])
            max_y = max([head.y, tail.y, max_y])
            min_z = min([head.z, tail.z, min_z])
            max_z = max([head.z, tail.z, max_z])

        b = first_head - first_tail
        b = b / torch.linalg.norm(b)

        pitch = torch.linalg.norm(torch.linalg.cross((second_head - first_head), b))
        print('Pitchmag:', pitch)

        #This becomes the pitch vector
        b = pitch * torch.tensor([b[2], b[1]])
        b = torch.linalg.matmul(
            b, torch.tensor([[0., -1.], [1., 0.]])
        )
        print('Pitch dir:', b)


        #Default: shift half a pitch lower so the rays bound a pitch-width centered on a wire
        if shift_half:

            #Decrement by half a pitch
            first_head = first_head[torch.tensor([2,1])]
            first_tail = first_tail[torch.tensor([2,1])]
            print(f'Original head/tail:\n\t{first_head}\n\t{first_tail}')
            first_head -= 0.5*b
            first_tail -= 0.5*b

            print(f'Shifted head/tail:\n\t{first_head}\n\t{first_tail}')

            #Now we have to account for the bounds in z and y
            first_head = torch.clamp(
                first_head,
                min=torch.tensor([min_z, min_y]),
                max=torch.tensor([max_z, max_y])
            )
            first_tail = torch.clamp(
                first_tail,
                min=torch.tensor([min_z, min_y]),
                max=torch.tensor([max_z, max_y])
            )
            print(f'New head/tail:\n\t{first_head}\n\t{first_tail}')

            first_center = torch.mean(torch.vstack([first_head, first_tail]), dim=0)
            print('FC:', first_center)



        second_point = first_center + b
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

def get_indices(coords, cross, i, j, k, round=False):
    #Find the locations in the third plane of crossing points from the first 2 planes
    #Then turn those into indices within the last plane
    base = len(coords.views) - 3
    locs = coords.pitch_location(i+base, cross[:, 0], j+base, cross[:, 1], k+base)
    # torch.save(locs, 'xover_locs.pt')
    indices = coords.pitch_index(locs, k+base, round=round)
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
