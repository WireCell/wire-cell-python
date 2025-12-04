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

def get_nwires(store, face_index=0):
    face = store.faces[face_index]
    nwires = [len(store.planes[i].wires) for i in face.planes]
    return nwires

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

def make_poly_sequence(coords, plane, nw, blobs_in, polys_in, run=1, warn=False):
    blobs_out = []
    polys_out = []
    plane_polys = {i:make_poly_insitu(coords, plane, i, i+run) for i in range(0, nw, run)}

    a = 0
    for poly, blob in zip(polys_in, blobs_in):
        b = 0
        print(a)
        for i, plane_poly_in in plane_polys.items():
            if not b % 10: print(b, end='\r')
            # print(i, plane_poly_in, poly, blob)
            new_poly = sutherland_hodgman_clip(plane_poly_in, poly)
            if new_poly.size(0) > 0: 
                polys_out.append(new_poly)
                blobs_out.append(torch.cat([blob, torch.tensor([[i, i+run]],dtype=blob.dtype)],dim=0))
            elif warn:
                print('WARNING SIZE 0 FROM BLOB')
            b += 1
        print()
        a += 1
    return polys_out, blobs_out

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
def shoelace_area(vertices: torch.Tensor) -> torch.Tensor:
    """
    Calculates the signed area of a polygon using the Shoelace formula.

    A positive result indicates a counter-clockwise orientation.
    A negative result indicates a clockwise orientation.

    Args:
        vertices (torch.Tensor): A tensor of shape (N, 2) where N is the number
                                   of vertices, and the second dimension holds (x, y)
                                   coordinates.

    Returns:
        torch.Tensor: A scalar tensor representing the signed area.
    """
    if vertices.dim() != 2 or vertices.size(1) != 2:
        raise ValueError("Vertices tensor must have shape (N, 2).")

    # 1. Separate coordinates
    x = vertices[:, 0]
    y = vertices[:, 1]

    # 2. Use torch.roll to get the coordinates of the next vertex (x_i+1, y_i+1)
    # The shift=-1 handles the wrap-around from N to 1.
    x_next = torch.roll(x, shifts=-1, dims=0)
    y_next = torch.roll(y, shifts=-1, dims=0)

    # 3. Calculate the two sums for the Shoelace formula:
    # S1 = sum(x_i * y_i+1)  (The "down/right" cross products)
    S1 = torch.sum(x * y_next)

    # S2 = sum(x_i+1 * y_i)  (The "up/left" cross products)
    S2 = torch.sum(x_next * y)

    # 4. Final Signed Area = 0.5 * (S1 - S2)
    signed_area = 0.5 * (S1 - S2)

    return abs(signed_area)
def make_cells(coords, nw_0, nw_1, nw_2, keep_shape=False):
    trivial_blobs = tiling.trivial_blobs()

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

def coords_from_schema(store, face_index, shift_half=True, left_handed=False):
    views = views_from_schema(store, face_index, shift_half=shift_half, left_handed=left_handed)
    coords = Coordinates(views.to(torch.float64))
    return coords

def get_center(store, wire):

    head = store.points[wire.head]
    tail = store.points[wire.tail]
    center =  torch.mean(torch.tensor([
            [head.z, head.y],
            [tail.z, tail.y],
    ], dtype=torch.float64), dim=0)
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
    ray_dir = coords.ray_dir[plane]
    xy = coords.views[plane][0] + ray_low*pitch_dir*pitch_mag
    p0 = (xy + extent*ray_dir)
    p1 = (xy - extent*ray_dir)
    p2 = p1 + (ray_high - ray_low)*pitch_dir*pitch_mag
    p3 = p0 + (ray_high - ray_low)*pitch_dir*pitch_mag

    #Ordered Clockwise
    poly = torch.vstack([p0.unsqueeze(0), p1.unsqueeze(0), p2.unsqueeze(0), p3.unsqueeze(0)])

    return poly

def is_inside(points, clip_edge_start, clip_edge_end):
    """
    Determines if points are 'inside' the half-plane defined by the clip edge.
    'Inside' is typically defined as the side where the cross product is positive (for counter-clockwise order).
    
    Args:
        points (torch.Tensor): Tensor of shape (N, 2) for subject vertices.
        clip_edge_start (torch.Tensor): (2,) for P_start of clip edge.
        clip_edge_end (torch.Tensor): (2,) for P_end of clip edge.
        
    Returns:
        torch.Tensor: Boolean tensor of shape (N,) where True means 'inside'.
    """
    # Vector from clip start to clip end: (C_end - C_start)
    edge_vec = clip_edge_end - clip_edge_start
    
    # Vector from clip start to subject points: (P - C_start)
    point_vecs = points - clip_edge_start
    
    # Cross product (2D equivalent: a_x * b_y - a_y * b_x)
    # The sign of the cross product determines which side of the line the point is on.
    cross_product = edge_vec[0] * point_vecs[:, 1] - edge_vec[1] * point_vecs[:, 0]
    
    # Assuming the clip polygon is ordered counter-clockwise (CCW), 
    # a positive cross product means the point is 'inside' the polygon.
    return cross_product >= 0

def find_intersection(p1, p2, c1, c2):
    """
    Finds the intersection point of two line segments (p1->p2 and c1->c2).
    
    Args:
        p1, p2 (torch.Tensor): Subject edge start/end, shape (2,).
        c1, c2 (torch.Tensor): Clip edge start/end, shape (2,).
        
    Returns:
        torch.Tensor: Intersection point (2,).
    """
    # Line 1 (Subject): P = p1 + t * (p2 - p1)
    # Line 2 (Clip): C = c1 + u * (c2 - c1)
    
    dp = p2 - p1
    dc = c2 - c1
    
    # Determinant D = dp_x * dc_y - dp_y * dc_x
    D = dp[0] * dc[1] - dp[1] * dc[0]
    
    # If D is close to zero, lines are parallel (or collinear)
    if torch.abs(D) < 1e-6:
        # Handle parallel/collinear case: for simplicity, we return the start point, 
        # but in a full implementation, you'd need more robust logic.
        return p1
    
    # Vector from clip start to subject start
    cp1 = p1 - c1
    
    # Solve for t (parameter along the subject edge)
    t = (cp1[1] * dc[0] - cp1[0] * dc[1]) / D
    
    # Intersection point
    intersection_point = p1 + t * dp
    
    return intersection_point

def sutherland_hodgman_clip(subject_polygon, clip_polygon):
    """
    Applies the Sutherland-Hodgman algorithm.
    
    Args:
        subject_polygon (torch.Tensor): (N_sub, 2)
        clip_polygon (torch.Tensor): (N_clip, 2) - MUST BE CONVEX and CCW ORDERED.
        
    Returns:
        torch.Tensor: The clipped polygon vertices (M, 2).
    """
    # Start with the original subject polygon
    output_polygon = subject_polygon.clone()
    
    # Get the number of vertices for the clip polygon
    num_clip_vertices = clip_polygon.shape[0]
    
    # 1. Iterate through each edge of the clip polygon
    for i in range(num_clip_vertices):
        # Clip edge: c1 -> c2
        c1 = clip_polygon[i]
        c2 = clip_polygon[(i + 1) % num_clip_vertices]
        
        input_polygon = output_polygon
        # If the polygon shrinks to 0 or 1 point, stop
        if input_polygon.shape[0] < 2:
            return torch.empty((0, 2)) 

        new_output_polygon = []
        num_input_vertices = input_polygon.shape[0]

        # 2. Iterate through each edge of the current subject polygon
        for j in range(num_input_vertices):
            # Subject edge: p1 -> p2
            p1 = input_polygon[j]
            p2 = input_polygon[(j + 1) % num_input_vertices]
            
            # Check the "inside" status of the endpoints (p1 and p2)
            p1_inside = is_inside(p1.unsqueeze(0), c1, c2).item()
            p2_inside = is_inside(p2.unsqueeze(0), c1, c2).item()
            
            # 3. Apply the Four Clipping Cases
            
            if p1_inside and p2_inside:
                # Case 1: Inside -> Inside (Output P2)
                new_output_polygon.append(p2)
            
            elif p1_inside and not p2_inside:
                # Case 2: Inside -> Outside (Output Intersection I)
                intersection = find_intersection(p1, p2, c1, c2)
                new_output_polygon.append(intersection)
                
            elif not p1_inside and p2_inside:
                # Case 4: Outside -> Inside (Output Intersection I, then P2)
                intersection = find_intersection(p1, p2, c1, c2)
                new_output_polygon.append(intersection)
                new_output_polygon.append(p2)
                
            # Case 3: Outside -> Outside (Output None) - implicitly handled by else/no-op

        # Update the subject polygon for the next clip edge
        if not new_output_polygon:
            output_polygon = torch.empty((0, 2))
        else:
            output_polygon = torch.stack(new_output_polygon)

    return output_polygon
def shoelace_area(vertices: torch.Tensor) -> torch.Tensor:
    """
    Calculates the signed area of a polygon using the Shoelace formula.

    A positive result indicates a counter-clockwise orientation.
    A negative result indicates a clockwise orientation.

    Args:
        vertices (torch.Tensor): A tensor of shape (N, 2) where N is the number
                                   of vertices, and the second dimension holds (x, y)
                                   coordinates.

    Returns:
        torch.Tensor: A scalar tensor representing the signed area.
    """
    if vertices.dim() != 2 or vertices.size(1) != 2:
        raise ValueError("Vertices tensor must have shape (N, 2).")

    # 1. Separate coordinates
    x = vertices[:, 0]
    y = vertices[:, 1]

    # 2. Use torch.roll to get the coordinates of the next vertex (x_i+1, y_i+1)
    # The shift=-1 handles the wrap-around from N to 1.
    x_next = torch.roll(x, shifts=-1, dims=0)
    y_next = torch.roll(y, shifts=-1, dims=0)

    # 3. Calculate the two sums for the Shoelace formula:
    # S1 = sum(x_i * y_i+1)  (The "down/right" cross products)
    S1 = torch.sum(x * y_next)

    # S2 = sum(x_i+1 * y_i)  (The "up/left" cross products)
    S2 = torch.sum(x_next * y)

    # 4. Final Signed Area = 0.5 * (S1 - S2)
    signed_area = 0.5 * (S1 - S2)

    return signed_area


def merge_close_points(points: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Merges points in a tensor that are closer than a specified epsilon 
    by replacing the cluster with its centroid.
    
    Args:
        points (torch.Tensor): Input tensor of shape (N, D), where N is the 
                               number of points and D is the dimension (e.g., 2).
        epsilon (float): The maximum distance threshold for points to be merged.
        
    Returns:
        torch.Tensor: The merged set of points (centroids).
    """
    # Use float64 for better precision in distance calculations
    points = points.to(torch.float64) 
    
    # Initialize the list for the final merged points
    merged_points = []
    
    # Create a mask to track which points have already been processed/merged
    is_merged = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    
    # Simple, sequential clustering loop (best done on CPU for sequential logic)
    for i in range(points.shape[0]):
        if is_merged[i]:
            continue
        
        # 1. Calculate Squared Distance: D_ij^2 = ||P_i - P_j||^2
        # Vectorized calculation of differences between the current point P_i and all others
        diff = points[i] - points
        
        # Squared Euclidean distance
        # Shape (N, D) -> (N,)
        sq_distances = torch.sum(diff ** 2, dim=1)
        
        # 2. Identify Neighbors: Find all points within the epsilon radius
        # Note: we use epsilon^2 to avoid the slow torch.sqrt() operation
        neighbors_mask = sq_distances <= (epsilon ** 2)
        
        # Get the actual points and indices that belong to this cluster
        cluster_points = points[neighbors_mask]
        
        # 3. Merge: Calculate the centroid (average) of the cluster
        # Shape (K, D) -> (D,)
        centroid = cluster_points.mean(dim=0)
        
        # 4. Update: Mark all points in this cluster as processed
        is_merged[neighbors_mask] = True
        
        # 5. Store the centroid
        merged_points.append(centroid)

    # Stack the centroids back into a single tensor
    if not merged_points:
        return torch.empty((0, points.shape[1]), dtype=points.dtype, device=points.device)
        
    return torch.stack(merged_points)

def get_inside_crossings(coords, blobs):
    xings = tiling.blob_crossings(blobs)
    insides = tiling.blob_insides(coords, blobs, xings)
    pairs = tiling.strip_pairs(blobs.shape[1])
    pairs = pairs.unsqueeze(0).unsqueeze(2).repeat(insides.shape[0], 1, insides.shape[2], 1)
    blob_inds = torch.arange(blobs.shape[0]).unsqueeze(1).unsqueeze(2).repeat(1, insides.shape[1], insides.shape[2])
    return dict(
        xings=xings[insides], pairs=pairs[insides], indices=blob_inds[insides], insides=insides, ncells=insides.shape[0]
    )


def merge_crossings(coords, inside_crossings, verbose=False):
    seen = 0
    centroids = []
    areas = []
    merged_cells = []
    inds = inside_crossings['indices']
    pairs = inside_crossings['pairs']
    xings = inside_crossings['xings']
    ncells = inside_crossings['ncells']
    rcs = coords.ray_crossing(pairs[:,0], xings[:,0], pairs[:,1], xings[:,1])
    unique_inds, counts = torch.unique(inds, return_counts=True)
    max_count = max(counts).item()
    for i in range(ncells):
    # for i in range(100):
        
        where = torch.where(inds[seen:seen+max_count] == i)
        vs_0 = rcs[seen:seen+max_count][where]
        if verbose and not i % 1000:
            print(i, where[0].shape, seen, end='\r')
        seen += where[0].shape[0]
        merged = merge_close_points(vs_0, 1.e-9)
        if len(merged) < 3: print('ERROR') #TODO -- throw
        centroid = torch.mean(merged, dim=0)
        centroids.append(centroid)
        diffs = centroid - merged
        angles = torch.atan2(diffs[:,1], diffs[:,0])
        _, sorted_inds = torch.sort(angles)
        areas.append(shoelace_area(merged[sorted_inds]))
        merged_cells.append(merged[sorted_inds])

    #Normalize area and centroids
    areas = torch.stack(areas).to(torch.float32)
    mean_area = torch.mean(areas)
    areas -= mean_area
    areas /= mean_area

    centroids = torch.stack(centroids).to(torch.float32)
    max_cent = torch.max(centroids, dim=0)
    min_cent = torch.min(centroids, dim=0)
    centroids -= (max_cent.values + min_cent.values)/2.0
    centroids *= (2./(max_cent.values - min_cent.values))

    return dict(
        crossings=merged_cells,
        centroids=centroids,
        areas=areas,
        mean_area=mean_area,
        max_centroid=max_cent.values,
        min_centroid=min_cent.values,
    )

def make_poly_from_blob(coords, blob, has_trivial=False):
    sub = 2 if has_trivial else 0
    polys = [make_poly_insitu(coords, i-sub, *blob[i]) for i in range(blob.shape[0])]
    poly = polys[0]
    for p in polys[1:]: poly = sutherland_hodgman_clip(poly, p)
    return poly



def make_all_poly(coords, blobs, nwires_in):
    poly_prims = []
    for i, nwires in enumerate(nwires_in):
        poly_prims.append([])
        for j in range(nwires):
            poly_prims[-1].append(
                make_poly_insitu(coords, i, j, j+1)
            )
    trivial_blob = make_poly_from_blob(coords, torch.tensor([[0,1], [0,1]], dtype=torch.long), has_trivial=True)
    polys = []
    for iblob, blob in enumerate(blobs):
        if not iblob % 1000: print(iblob, end='\r')
        i = blob[2, 0]
        j = blob[3, 0]
        k = blob[4, 0]
        these_polys = [poly_prims[0][i], poly_prims[1][j], poly_prims[2][k]]
        polys.append(trivial_blob)
        for poly in these_polys:
            polys[-1] = sutherland_hodgman_clip(polys[-1], poly)
        if polys[-1].size(0) == 0: print('Empty', i, j, k, blob)

def get_centroids_and_areas(coords, blobs):


    areas = []
    centroids = []
    trivial_blob = make_poly_from_blob(coords, torch.tensor([[0,1], [0,1]], dtype=torch.long), has_trivial=True)
    for i, blob in enumerate(blobs):
        if not (i % 1000): print(i, end='\r')
        poly = make_poly_from_blob(coords, blob[2:], has_trivial=False)
        poly = sutherland_hodgman_clip(poly, trivial_blob)
        # areas.append(shoelace_area(poly))
        # centroids.append(torch.mean(poly, dim=0))
    areas = torch.vstack(areas)
    centroids = torch.cat(centroids)
    return areas, centroids       


def views_from_schema(store, face_index, shift_half=True, left_handed=False):
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

        first_head =  torch.tensor([first_head.x, first_head.y, first_head.z], dtype=torch.float64)
        first_tail =  torch.tensor([first_tail.x, first_tail.y, first_tail.z], dtype=torch.float64)
        second_head = torch.tensor([second_head.x, second_head.y, second_head.z], dtype=torch.float64)
        second_tail = torch.tensor([second_tail.x, second_tail.y, second_tail.z], dtype=torch.float64)
       

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
        # print('Pitchmag:', pitch)

        #This becomes the pitch vector
        b = pitch * torch.tensor([b[2], b[1]], dtype=torch.float64)
        if left_handed:
            b = torch.linalg.matmul(
                b, torch.tensor([[0., 1.], [-1., 0.]], dtype=torch.float64)
            )
        else:
            b = torch.linalg.matmul(
                b, torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64)
            )
        # print('Pitch dir:', b)


        #Default: shift half a pitch lower so the rays bound a pitch-width centered on a wire
        if shift_half:

            #Decrement by half a pitch
            first_head = first_head[torch.tensor([2,1])]
            first_tail = first_tail[torch.tensor([2,1])]
            # print(f'Original head/tail:\n\t{first_head}\n\t{first_tail}')
            shift = 0.5*b
            shifted_head = first_head - shift
            shifted_tail = first_tail - shift

            # print(f'Shifted head/tail:\n\t{first_head}\n\t{first_tail}')

            #Now we have to account for the bounds in z and y
            clamped_head = torch.clamp(
                shifted_head,
                min=torch.tensor([min_z, min_y]),
                max=torch.tensor([max_z, max_y])
            )
            clamped_tail = torch.clamp(
                shifted_tail,
                min=torch.tensor([min_z, min_y]),
                max=torch.tensor([max_z, max_y])
            )
            # print(f'Clamped head/tail:\n\t{clamped_head}\n\t{clamped_tail}')


            first_to_clamped_head = clamped_head - first_head
            first_to_clamped_tail = clamped_tail - first_tail

            ftc_unit_head = first_to_clamped_head / first_to_clamped_head.norm()
            ftc_unit_tail = first_to_clamped_tail / first_to_clamped_tail.norm()

            clamped_to_shifted_head = shifted_head - clamped_head
            clamped_to_shifted_tail = shifted_tail - clamped_tail

            big_v_head_norm = clamped_to_shifted_head.norm()**2 / first_to_clamped_head.norm()
            big_v_tail_norm = clamped_to_shifted_tail.norm()**2 / first_to_clamped_tail.norm()

            final_head = clamped_head + ftc_unit_head*big_v_head_norm
            final_tail = clamped_tail + ftc_unit_tail*big_v_tail_norm

            first_center = torch.mean(torch.vstack([final_head, final_tail]), dim=0)
            # print('FC:', first_center)



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
