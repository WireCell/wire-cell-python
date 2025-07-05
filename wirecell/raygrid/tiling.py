#!/usr/bin/env python

import torch
from .activity import threshold_1d

from collections import namedtuple
from itertools import product, combinations

# represent the state of a tiling.
class Tiling:
    def __init__(self, blobs, crossings, insides):

        # (N blobs, N views, 2 ray indices)
        self.blobs = blobs
        # (N blobs, N strip pairs, 4 strip edge pairs, 2 views, 2 rays)
        self.crossings = crossings
        # (N blobs, N strip pairs, 4 strip edge pairs)
        self.insides = insides

    @property
    def nviews(self):
        '''
        Number of view layers of this tiling
        '''
        return self.blobs.shape[1]

    @property
    def nblobs(self):
        '''
        Number of view layers of this tiling
        '''
        return self.blobs.shape[0]

    def select(self, ind):
        '''
        Return new tiling with a subset of blobs given by indices.
        '''
        return Tiling(self.blobs[ind],
                      self.crossings[ind],
                      self.insides[ind])

def trivial():
    '''
    Return tiling solution for trivial 2 view case
    '''

    # A "blob" is a 2D tensor of shape (N, 2).  The first dimension of size N is
    # the number of views and equivalently the number of defining strips.  The
    # second dimension of size 2 gives the strip bounds in its view.  The plural
    # "blobs" is simply a batch of blob tensors.
    # 
    # shape: (blob in set, strip in blob, bound in strip)
    blobs = torch.tensor([ [ [0,1], [0,1] ] ], dtype=torch.long )

    # A single "crossing" is a 2D tensor of shape (2,2).  The first dimension
    # runs over the two raygrid ind indices, one for each ray.  That is,
    # crossing[0] is the first raygrid index.  A raygrid index is a 2-tensor:
    # (layer, ray) of scalar indices.  Any pair of strips have 4 edge-pair-wise
    # crossings.  For one blob of N layers, there is N-choose-2 pairs of strips.
    # The plural "crossings" implies multiple blobs of multiple views.
    #
    # The trivial crossings are constructed in a hard-wired fashion.  First, the
    # 4 ray crossings of the pair of trivial strips:
    #
    # Enumerate the raygrid indices for the 4 edges of two strips
    strip_pair = torch.tensor(list(product( (0,1), (0,1))), dtype=torch.long)
    # Select all possible crossings.
    crossings = strip_pair[torch.tensor([(0, 2), (0, 3), (1, 2), (1, 3)])]
    # Reshape to give (1) blob and (1) strip pair dimensinos
    # (nblobs, stip-pair-in-blob, crossings, view, ray)
    # (nblobs=1, C(2,2)=1, 4, 2, 2)
    crossings = crossings.reshape((1,1,4,2,2))

    # The "inside" tensor matches crossings but instead of a 2x2 leaf we have a
    # scalar boolean for each blob and crossing in the blob.  True is
    # interpreted to mean the crossing is "in" the blob and thus a corner.
    #
    # Dereferencing like crossings[inside] gives a shape (N,2,2) of crossings.
    insides = torch.ones((1,1,4), dtype=torch.bool)

    return Tiling(blobs, crossings, insides)
    

def crossings_flatten(c):
    '''
    Given a 5D crossings tensor c of shape (Nblobs, Npairs, 4, 2, 2) return
    a 3d tensor of shape (2,2, Nblobs*Npairs*4).

    To unflatten, call .view(c.shape) on the returned flattened tensor.

    If the (2,2,N) is reduced to (N,) the result can be restored to pair level
    with .view(c.shape[:3]).
    '''
    return c.view(-1, 2, 2).permute(1, 2, 0)
    


def projection(coords, tiling):
    '''
    Return bounds of blobs projected into a new layer

    - coords :: A raygrid.Coordinates 
    - tiling :: A Tiling

    Bounds are returned as tuple of tensor (lo, hi).  The tensors are 1D of
    shape (Nblobs,) The pair give the per-blob half-open bounds.  Only blobs at
    index where lo < hi have finite size.  Where lo may have value positive
    infinity or hi has negative infinity then the corresponding blobs had no
    corners to start.  This can be avoided by calling tiling.select() on input.
    Also lo and hi should be checked for being in the bounds of the view.
    '''
    # restrict to crossings that are inside
    ins = tiling.insides.view(-1)
    fc = crossings_flatten(tiling.crossings) # (2,2,Nblobs*Npairs*4)

    # Select just the crossings that are also corners to reduce the next
    # calculation.
    fcins = fc[:,:,ins]

    # The pitches of the crossings in the new layer
    next_layer = tiling.nviews
    pins = coords.pitch_location(fcins[0], fcins[1], next_layer)
    
    # Find per-blob minimum pitch.  To do this we re-inflate the corner pitches
    # to (Nblobs*Npairs*4), with default value of positive infinity, set only
    # the corners to their pitches and then reshape to (Nblobs, Npairs*4).
    pmin = torch.zeros(fc[0,0].shape)
    pmin[:] = torch.inf
    pmin[ins] = pins
    pmin = pmin.view(tiling.nblobs, -1)
    vmin, imin = torch.min(pmin, dim=1)
    lo = torch.floor(imin).int()

    # Same but for max / hi
    pmax = torch.zeros(fc[0,0].shape)
    pmax[:] = -torch.inf
    pmax[ins] = pins
    pmax = pmax.view(tiling.nblobs, -1)
    vmax, imax = torch.max(pmax, dim=1)
    hi = torch.ceil(imax).int()
                    
    return (lo, hi)


def blob_edges(lo, hi, padded_active):
    '''
    Return edges of activity between half-open [lo,hi) range on padded_active.

    The padded_active is bool and True where there is view activity.  It must
    have one extra False element before and after the actual view domain.  That
    is, the view domain is 2 elements smaller.

    The [lo,hi) half-open range must be in view domain and must not consider
    this extra padding.

    The returned edges will extend by one element longer than the view domain.
    It is +1 at the start of an active region bounded by [lo,hi) and -1 at the
    element past the last element of an active region.

    (This is not a batched function.)
    '''
    padded_len = padded_active.numel()

    # clamp bounds to be physical to the unpadded length
    alen = padded_len - 2
    if lo < 0:
        lo = 0
    if hi > alen:
        hi = alen
    if lo >= hi:
        return

    mask = torch.full((padded_len,), False, dtype=torch.bool)
    mask[lo + 1 : hi + 1] = True

    edges = (padded_active & mask).int().diff()

    if not torch.any(edges):
        return

    return edges
    

def extend_blob(old_blob, edges):
    '''
    Extend a single blob to multiple blobs by splitting up via edges.

    Edges are as returned by blob_edges().
    '''
    print(f'{edges=}')
    begs = (edges == 1).nonzero(as_tuple=True)[0]
    ends = (edges == -1).nonzero(as_tuple=True)[0]

    nseg = begs.numel()
    print(f'{nseg=} {begs=} {ends=}')
    if nseg == 0:
        return

    # Replicate existing blob, to form basis for each new on
    blob_copies = old_blob.unsqueeze(0).expand(nseg, -1, -1)

    # make new strips, one per each new blob, shaped for blob tensor
    strips = torch.stack((begs, ends), dim=1).unsqueeze(1)

    # form new blobs
    blobs = torch.cat((blob_copies, strips), dim=1)
    
    return blobs


# totally written by gemini 2.5 flash
def fresh_insides(blobs: torch.Tensor, crossings: torch.Tensor) -> torch.Tensor:
    """
    Determines if specified crossing points are "inside" all strips of their respective blobs.

    A crossing point is considered "inside" all strips of its blob if both of its
    constituent values (derived from the specified strip and edge indices) fall
    within the inclusive [start, end] range of *every* strip within that blob.

    Args:
        blobs (torch.Tensor): A tensor of shape (nblobs, nstrips, 2).
                              The last dimension defines the inclusive [start, end]
                              range for each strip.
                              Example: `blobs[b, s, 0]` is the start of strip `s` for blob `b`,
                                       `blobs[b, s, 1]` is the end of strip `s` for blob `b`.

        crossings (torch.Tensor): A tensor of shape (nblobs, npairs, 4, 2, 2).
                                  - The first dimension (nblobs) corresponds to the blob index.
                                  - The second dimension (npairs) corresponds to a pair of strips.
                                  - The third dimension (4) spans the four combinations of
                                    low/high bounds for the pair of strips:
                                    [(lo1,lo2), (lo1, hi2), (hi1, lo2), (hi1, hi2)].
                                  - The fourth dimension (2) represents the two points that
                                    make up a crossing.
                                  - The fifth dimension (2) contains:
                                    - Index 0: The strip index within `blobs`.
                                    - Index 1: The edge index (0 for 'lo'/'start', 1 for 'hi'/'end').
                                  Example: `crossings[b, p, c, 0, 0]` is the strip index for point 1,
                                           `crossings[b, p, c, 0, 1]` is the edge index for point 1.

    Returns:
        torch.Tensor: A boolean tensor of shape (nblobs, npairs, 4).
                      `True` indicates that the corresponding crossing point (defined by
                      both its constituent values) is inside all strips of its blob.
    """
    # Extract dimensions for clarity and broadcasting
    nblobs, nstrips, _ = blobs.shape
    _, npairs, ncombinations, _, _ = crossings.shape # ncombinations is always 4

    print (f'{nblobs=} {npairs=} {ncombinations=}')

    # 1. Extract the strip indices and edge indices for the two points (P1 and P2)
    # of each crossing. These will have shape (nblobs, npairs, 4).
    strip_indices_p1 = crossings[..., 0, 0]
    edge_indices_p1 = crossings[..., 0, 1]
    strip_indices_p2 = crossings[..., 1, 0]
    edge_indices_p2 = crossings[..., 1, 1]

    # 2. Flatten the 'blobs' tensor to easily gather the actual values of the points.
    # `blobs_flat_per_blob` will have shape (nblobs, nstrips * 2), where each row
    # contains all start/end values for a single blob, flattened.
    blobs_flat_per_blob = blobs.view(nblobs, -1)

    # 3. Create flat indices to gather the actual numerical values of the crossing points.
    # For each point (P1 or P2), its value is `blobs[blob_idx, strip_index, edge_index]`.
    # In the flattened `blobs_flat_per_blob`, the index for `blobs[b, s, e]` is `s * 2 + e`.
    # These flat indices will have shape (nblobs, npairs, 4).
    flat_indices_p1 = strip_indices_p1 * 2 + edge_indices_p1
    flat_indices_p2 = strip_indices_p2 * 2 + edge_indices_p2

    # Reshape the flat indices to (nblobs, npairs * 4) to match the `torch.gather` requirements.
    flat_indices_p1_reshaped = flat_indices_p1.view(nblobs, -1)
    flat_indices_p2_reshaped = flat_indices_p2.view(nblobs, -1)

    # Gather the actual numerical values of the crossing points.
    # `val_p1` and `val_p2` will initially have shape (nblobs, npairs * 4) after `gather`,
    # then they are reshaped back to (nblobs, npairs, 4) to represent the value for
    # each (blob, pair, combination).
    val_p1 = torch.gather(blobs_flat_per_blob, 1, flat_indices_p1_reshaped).view(nblobs, npairs, ncombinations)
    val_p2 = torch.gather(blobs_flat_per_blob, 1, flat_indices_p2_reshaped).view(nblobs, npairs, ncombinations)

    # 4. Prepare the strip start and end values for efficient broadcasting.
    # We expand `strip_starts` and `strip_ends` to (nblobs, 1, 1, nstrips).
    # The `unsqueeze` operations add singleton dimensions that allow `val_p1_expanded`
    # and `val_p2_expanded` (which will be (nblobs, npairs, 4, 1)) to broadcast
    # correctly across all `nstrips` for each check.
    strip_starts = blobs[:, :, 0].unsqueeze(1).unsqueeze(1)
    strip_ends = blobs[:, :, 1].unsqueeze(1).unsqueeze(1)

    # Expand `val_p1` and `val_p2` to (nblobs, npairs, 4, 1) for broadcasting.
    val_p1_expanded = val_p1.unsqueeze(-1)
    val_p2_expanded = val_p2.unsqueeze(-1)

    # 5. Check if `val_p1` is inside each individual strip.
    # This results in a boolean tensor of shape (nblobs, npairs, 4, nstrips).
    is_p1_ge_start = (val_p1_expanded >= strip_starts) # Is P1 value >= start of each strip?
    is_p1_le_end = (val_p1_expanded <= strip_ends)     # Is P1 value <= end of each strip?
    is_p1_inside_each_strip = is_p1_ge_start & is_p1_le_end # Is P1 value inside each strip's range?

    # For `val_p1` to be "inside all strips", it must be inside *every* strip in its blob.
    # `.all(dim=-1)` checks this condition across the `nstrips` dimension,
    # resulting in a boolean tensor of shape (nblobs, npairs, 4).
    is_p1_inside_all_strips = is_p1_inside_each_strip.all(dim=-1)

    # 6. Perform the same "inside all strips" check for `val_p2`.
    is_p2_ge_start = (val_p2_expanded >= strip_starts)
    is_p2_le_end = (val_p2_expanded <= strip_ends)
    is_p2_inside_each_strip = is_p2_ge_start & is_p2_le_end
    is_p2_inside_all_strips = is_p2_inside_each_strip.all(dim=-1)

    # 7. A crossing is considered "inside" if and only if both of its constituent
    # points (`val_p1` and `val_p2`) are inside all strips of the blob.
    inside = is_p1_inside_all_strips & is_p2_inside_all_strips

    return inside

    

def fresh_crossings(blobs, old_crossings):
    '''
    Given blobs made from a single old blob and that old blob's crossings,
    return fresh crossings spanning new blobs.
    '''
    nblobs = blobs.shape[0]
    nprior = blobs.shape[1] - 1

    # 1. Generate combination of old views and new view
    pair_indices = torch.tensor(list(product(range(nprior), [nprior])))
    npairs = pair_indices.shape[0]
    print(f"  {npairs=} {pair_indices=}")

    # 2. Extract paired data for all nblobs new blobs simultaneously
    # data_for_idx1_b_prime will be (nblobs, npairs_b_prime, 2)
    data_for_idx1 = blobs[:, pair_indices[:, 0], :]
    data_for_idx2 = blobs[:, pair_indices[:, 1], :]
    print(f'{data_for_idx1=}')
    print(f'{data_for_idx2=}')


    # 3. Construct the 4 combinations of values ((lo,hi) x (lo,hi))
    lo1 = data_for_idx1[..., 0] # (nblobs, npairs)
    hi1 = data_for_idx1[..., 1] # (nblobs, npairs)
    lo2 = data_for_idx2[..., 0] # (nblobs, npairs)
    hi2 = data_for_idx2[..., 1] # (nblobs, npairs)

    # Stack the 4 value combinations
    # Each stack creates a new last dimension of size 2
    lo1_lo2_val = torch.stack([lo1, lo2], dim=-1) # (nblobs, npairs, 2)
    lo1_hi2_val = torch.stack([lo1, hi2], dim=-1)
    hi1_lo2_val = torch.stack([hi1, lo2], dim=-1)
    hi1_hi2_val = torch.stack([hi1, hi2], dim=-1)

    # Stack these 4 to get the (nblobs, npairs, 4, 2) tensor of values
    value_combinations_for_blobs = torch.stack(
        [lo1_lo2_val, lo1_hi2_val, hi1_lo2_val, hi1_hi2_val],
        dim=-2 # Stack along the 3rd to last dimension
    )
    print(f"  Value combinations shape: {value_combinations_for_blobs.shape=}")

    # 4. Construct the corresponding layer_idx combinations
    # pair_indices is (npairs, 2)
    # We need to expand it to (nblobs, npairs, 4, 2)
    layer_idx_combinations_for_blobs = pair_indices.unsqueeze(0).unsqueeze(2).expand(
        nblobs, -1, 4, -1
    )

    print(f"  Layer_idx combinations shape: {layer_idx_combinations_for_blobs.shape}")

    # 5. Final Concatenation to get (nblobs, npairs, 4, 2, 2)
    # Unsqueeze value and layer_idx combos to (nblobs, npairs, 4, 1, 2)
    value_combinations_final_reshaped = value_combinations_for_blobs.unsqueeze(-2)
    layer_idx_combinations_final_reshaped = layer_idx_combinations_for_blobs.unsqueeze(-2)

    new_crossings = torch.cat(
        (layer_idx_combinations_final_reshaped, value_combinations_final_reshaped),
        dim=-2 # Concatenate along the second to last dimension
    )
    print(f"  {new_crossings.shape=}")
    
    crossings_copies = old_crossings.unsqueeze(0).expand(nblobs, -1, -1, -1, -1)
    print(f"  {crossings_copies.shape=}")

    crossings = torch.cat((crossings_copies, new_crossings), dim=1)
    return crossings
    

def apply_view(coords, prior, active):
    '''
    Return a new Tiling with view's active measures applied to prior Tiling.
    '''

    # First, find the per-blob bounds in the next view.
    los, his = projection(coords, prior)

    # Remove any out-of-domain blobs.  Reminder: hi is one past so it can be
    # just out of domain.
    n = active.numel()
    inds = (los < his) & (los >=0) & (his > 0) & (los < n) & (his <= n)
    keep = prior.select(inds)

    #
    # Prepare to find new blobs
    # 

    # pad so any True at start/end has a level crossing.
    padded_active = torch.cat((
        torch.tensor([False]),
        active,
        torch.tensor([False])
    ))

    collect_blobs = list()
    collect_crossings = list()
    collect_insides = list()


    # Next, we derive a new blob from each prior blob.  This has to be an
    # explicit loop as its a one-to-many expansion.
    for ind in range(keep.nblobs):
        edges = blob_edges(los[ind], his[ind], padded_active)
        if edges is None:
            continue

        blobs = extend_blob(keep.blobs[ind], edges)
        collect_blobs.append(blobs)

        crossings = fresh_crossings(blobs, keep.crossings[ind])
        collect_crossings.append(crossings)


    if not collect_blobs:
        return 

    all_blobs = torch.cat(collect_blobs, dim=0)
    all_crossings = torch.cat(collect_crossings, dim=0)

    return Tiling(all_blobs, all_crossings,
                  fresh_insides(all_blobs, all_crossings));
