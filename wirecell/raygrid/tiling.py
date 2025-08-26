#!/usr/bin/env python

import torch
from .activity import threshold_1d

from collections import namedtuple
from itertools import product, combinations

# Represent the state of a tiling.  This class is not actually used in most
# function which instead take individual tensors.  However, the tensor names
# described here are relevant throughout.
class Tiling:
    def __init__(self, blobs, crossings, insides):

        # The blobs tensor.
        #
        # (N blobs, N strips, 2 ray indices)
        #
        # The 2 ray indices are give bounds in the ray grid array.
        self.blobs = blobs

        # The crossing tensor.
        #
        # (N blobs, N strip pairs, 4 strip edge pairs, 2 strips, 2 rays)
        #
        # This enumerates pairs of strips in a blob and pairs of edges of each
        # pair of strip.  The last two dimensions give indices into the last two
        # dimensions of a blob tensor.  Specifically, the last index is NOT a
        # ray grid array index.  One must always use the blobs tensor to go back
        # to a ray grid array index.
        self.crossings = crossings

        # The insides tensor.
        #
        # (N blobs, N strip pairs, 4 strip edge pairs)
        #
        # This is a boolean tensor that is true if a strip-edge pair is inside
        # the blob bounds.
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

    def as_string(self, depth=1):
        lines = [f'blobs shape: {self.blobs.shape}']
        if depth > 1:
            lines += [f'\t{self.blobs=}']
        if depth > 2:
            lines += [f'\t{self.crossings=}']

        return '\n'.join(lines)

def strip_pairs(nviews):
    '''
    Return a blob strip pair view index tensor for a blob of the given
    number of views.

    Return shape (npairs, 2)

    Value is view/strip index.

    Note, this returns a stable order so that strip_pairs(nviews) is the first
    part of strip_pairs(nviews+1).
    '''
    if nviews < 2:
        raise ValueError(f'too few views to make pairs: {nviews}')
    if nviews == 2:
        return torch.tensor([[0, 1]], dtype=torch.long)

    prior = strip_pairs(nviews-1)
    more = torch.tensor(list(product(range(nviews-1), [nviews-1])))
    return torch.vstack( (prior, more) )


def expand_first(ten, n):
    '''
    Replicate tensor n-times along a new first dimension
    '''
    shape = [n] + [-1] * ten.ndim
    return ten.unsqueeze(0).expand(*shape)

    
def strip_pair_edge_indices():
    '''
    When two strips overlap, there are 4 edge crossings.  This function
    returns a (4,2) tensor giving the canonical ordering of the 4 crossings of
    pairs of edges.  The first dimension indexes one of possible 4
    strip-edge-paring.  The last dimension is the edge index (0=lo, 1=hi).

    The edge index may be used, for example, in the last dimension of a blobs
    tensor.

    '''
    return torch.tensor([ [0,0], [0,1], [1,0], [1,1] ])


def blob_crossings(blobs):
    '''
    Return a blob crossings tensor.

    This is shape (nblobs, npairs, 4, 2) and type integer holding ray indices.

    Dimensions:

    0. run over blobs
    1. run over nstrips-choose-2 pairs of strips
    2. run over the 4 pairs of edges of 2 strips (see strip_pair_edge_indices() for order)
    3. the raygrid RAY index for each edge ray (eg, values from the last dimension of blobs)

    '''
    nstrips = blobs.shape[1]
    # (npairs)
    views_a, views_b = strip_pairs(nstrips).T

    edges_a, edges_b = strip_pair_edge_indices().T

    # (nblobs, npairs, 4)
    rays_a = blobs[:, views_a, :][..., edges_a]
    rays_b = blobs[:, views_b, :][..., edges_b]
    return torch.stack((rays_a, rays_b), dim=3)
    

def blob_strip_pairs(blobs):
    '''
    Return a blob strip pairs tensor

    This is shape (nblobs, npairs, 4, 2) and type integer holding view indices.

    Dimensions:

    0. run over blobs
    1. run over nstrips-choose-2 pairs of strips
    2. run over the 4 pairs of edges of 2 strips (see strip_pair_edge_indices() for order)
    3. the raygrid VIEW index for each stripo (eg, values from the second dimension of blobs)

    This is the expansion of strip_pairs() to the proper shape.
    '''
    pass
    

def blob_insides(coords, blobs, crossings):
    """
    Constructs the (nblobs, npairs, 4) 'insides' tensor.
    Each element is True if the corresponding crossing point is inside
    ALL other strips of the blob.

    Args:
        coords: The coordinate transformation object.
        blobs: The (nblobs, nviews, 2) tensor.
        crossings: The (nblobs, npairs, 4, 2) tensor.

    Returns:
        A (nblobs, npairs, 4) boolean torch.Tensor indicating 'insideness'.
    """
    nblobs, nviews, _ = blobs.shape
    _, npairs, num_crossing_types, _ = crossings.shape # num_crossing_types is 4

    if nviews < 3:
        # If nviews < 3, there are no 'other' strips to check against (nviews-2 is < 1).
        # In this case, all crossings are trivially "inside" all other strips (because there are none).
        return torch.ones((nblobs, npairs, num_crossing_types), dtype=torch.bool, device=blobs.device)

    # 1. Prepare base flattened crossing data (v1, r1, v2, r2)
    # The 'nviews' here is the current total number of views the 'crossings' tensor was built with.
    v1_base, r1_base, v2_base, r2_base = flatten_crossings(crossings, nviews)
    
    # These are of shape (nblobs * npairs * 4,)
    num_flattened_crossings = v1_base.numel()

    # 2. Determine 'other' view indices for each pair
    # (npairs, 2)
    current_pairs = strip_pairs(nviews)
    
    # all_view_indices: (nviews,) e.g., [0, 1, 2, ..., nviews-1]
    all_view_indices = torch.arange(nviews, dtype=torch.long, device=blobs.device)

    # (npairs, nviews) boolean mask, True where view is one of the pair views
    is_in_pair_mask = (all_view_indices.unsqueeze(0) == current_pairs[:, 0].unsqueeze(1)) | \
                      (all_view_indices.unsqueeze(0) == current_pairs[:, 1].unsqueeze(1))

    # (npairs, nviews) boolean mask, True where view is NOT one of the pair views
    is_other_view_mask = ~is_in_pair_mask

    # List of 1D tensors, each holding the 'other' view indices for a specific pair
    # Each list element will be (n_others_per_pair,)
    other_views_list = [all_view_indices[is_other_view_mask[p_idx]] for p_idx in range(npairs)]

    # We need to stack these for each pair.
    # The number of 'other' views is consistent for all pairs: nviews - 2.
    n_others_per_pair = nviews - 2
    
    # Stack 'other' views into a (npairs, n_others_per_pair) tensor
    other_views_tensor = torch.stack(other_views_list, dim=0) # (npairs, n_others_per_pair)

    # 3. Prepare full flattened inputs for crossing_in_other
    # We need to combine the (nblobs*npairs*4) dimension with the (n_others_per_pair) dimension.
    # Total effective calls: nblobs * npairs * 4 * n_others_per_pair

    # Reshape base crossing inputs to allow broadcasting with other_views
    # (nblobs*npairs*4,) -> (nblobs*npairs*4, 1)
    v1_expanded = v1_base.unsqueeze(1)
    r1_expanded = r1_base.unsqueeze(1)
    v2_expanded = v2_base.unsqueeze(1)
    r2_expanded = r2_base.unsqueeze(1)

    # v3: Each (nblobs*npairs*4) crossing needs to be checked against each of its relevant 'other' views.
    # We need to map other_views_tensor (npairs, n_others_per_pair)
    # to (nblobs*npairs*4, n_others_per_pair)
    
    # Replicate other_views_tensor for each crossing of a specific pair
    # First, repeat for the 4 crossing types: (npairs, n_others_per_pair) -> (npairs, 4, n_others_per_pair)
    v3_base_per_pair_and_crossing = other_views_tensor.unsqueeze(1).expand(-1, num_crossing_types, -1)
    
    # Then, reshape to (npairs*4, n_others_per_pair)
    v3_base_per_crossing = v3_base_per_pair_and_crossing.reshape(npairs * num_crossing_types, n_others_per_pair)

    # Finally, repeat for each blob: (nblobs, npairs*4, n_others_per_pair)
    v3_flat = v3_base_per_crossing.unsqueeze(0).expand(nblobs, -1, -1).reshape(-1) # (nblobs * npairs * 4 * n_others_per_pair,)

    # rbegin3 and rend3: these come from blobs[:, v3, 0/1]
    # We need (nblobs, n_others_per_pair) for each blob's relevant strip bounds.
    # blobs[:, v3, 0]: (nblobs, n_others_per_pair)
    
    # For each blob, we gather the relevant bounds for its 'other' strips.
    # blobs_lo_others: (nblobs, npairs, n_others_per_pair) - this is tricky because v3 depends on npairs.
    # Simpler: gather all blob bounds (nblobs, nviews, 2)
    # Then, use v3_flat to index into this.
    
    # Create the full index for `blobs`
    # blob_idx_for_r3: (nblobs * npairs * 4 * n_others_per_pair,)
    # It needs to repeat each blob_idx for all (npairs * 4 * n_others_per_pair) checks.
    
    blob_indices_overall = torch.arange(nblobs, dtype=torch.long) #  fixme device
    # Repeat each blob index for all (npairs * 4 * n_others_per_pair) tests
    # Shape: (nblobs, npairs * 4 * n_others_per_pair) -> (nblobs * npairs * 4 * n_others_per_pair,)
    blob_indices_flat = blob_indices_overall.unsqueeze(1).expand(-1, npairs * num_crossing_types * n_others_per_pair).reshape(-1)

    # Get rbegin3 and rend3
    # Use advanced indexing to get specific blob, specific other view (v3_flat), specific bound (0 or 1).
    # rbegin3_flat: (nblobs * npairs * 4 * n_others_per_pair,)
    # rend3_flat: (nblobs * npairs * 4 * n_others_per_pair,)
    rbegin3_flat = blobs[blob_indices_flat, v3_flat, 0]
    rend3_flat = blobs[blob_indices_flat, v3_flat, 1]

    # Expand v1, r1, v2, r2 to match the total flattened size
    # (nblobs*npairs*4,) -> (nblobs*npairs*4, n_others_per_pair) -> (nblobs*npairs*4*n_others_per_pair,)
    v1_final = v1_expanded.expand(-1, n_others_per_pair).reshape(-1)
    r1_final = r1_expanded.expand(-1, n_others_per_pair).reshape(-1)
    v2_final = v2_expanded.expand(-1, n_others_per_pair).reshape(-1)
    r2_final = r2_expanded.expand(-1, n_others_per_pair).reshape(-1)

    # 4. Call crossing_in_other with all flattened inputs
    # check_results: (nblobs * npairs * 4 * n_others_per_pair,) boolean tensor
    check_results = crossing_in_other(
        coords,
        v1_final, r1_final,
        v2_final, r2_final,
        v3_flat,
        rbegin3_flat, rend3_flat
    )

    # 5. Reshape results back
    # (nblobs * npairs * 4 * n_others_per_pair,) -> (nblobs, npairs, 4, n_others_per_pair)
    reshaped_check_results = check_results.reshape(nblobs, npairs, num_crossing_types, n_others_per_pair)

    # 6. Combine results: A crossing is "inside" if it's inside ALL other strips.
    # Perform a logical AND reduction along the 'n_others_per_pair' dimension.
    # insides: (nblobs, npairs, 4)
    insides = torch.all(reshaped_check_results, dim=-1)

    return insides



def trivial_blobs():
    return torch.tensor([ [ [0,1], [0,1] ] ], dtype=torch.long )    

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
    blobs = trivial_blobs()

    # A single "crossing" is a 2D tensor of shape (2,2).  The first dimension
    # runs over the two raygrid ind indices, one for each ray.  That is,
    # crossing[0] is the first raygrid index.  A raygrid index is a 2-tensor:
    # (layer, ray) of scalar indices.  Any pair of strips have 4 edge-pair-wise
    # crossings.  For one blob of N layers, there is N-choose-2 pairs of strips.
    # The plural "crossings" implies multiple blobs of multiple views.
    #
    # (nblobs=1, C(2,2)=1, 4, 2, 2)
    crossings = blob_crossings(blobs)

    # The "inside" tensor matches crossings but instead of a 2x2 leaf we have a
    # scalar boolean for each blob and crossing in the blob.  True is
    # interpreted to mean the crossing is "in" the blob and thus a corner.
    #
    # Dereferencing like crossings[inside] gives a shape (N,2,2) of crossings.
    #
    # Non-trivial case, call blob_insides() but the trivial case can be made
    # without requiring a coords object.
    insides = torch.ones((1,1,4), dtype=torch.bool)

    return Tiling(blobs, crossings, insides)
    
def flatten_crossings(crossings, nviews):
    '''
    nviews is the number of views from current crossings.  FIXME: can derive
    that from crossings.shape[1].
    '''
    nblobs = crossings.shape[0]
    npairs = crossings.shape[1]

    # Validate crossings shape
    expected_crossings_shape = (nblobs, npairs, 4, 2)
    if crossings.shape != expected_crossings_shape:
        raise ValueError(
            f"crossings tensor has unexpected shape: {crossings.shape}. "
            f"Expected {expected_crossings_shape} for nblobs={nblobs}, nviews={nviews}."
        )

    # 2. Extract r1 and r2 directly from crossings
    # Flatten across nblobs, npairs, and 4
    r1 = crossings[..., 0].reshape(-1) # Shape (nblobs * npairs * 4,)
    r2 = crossings[..., 1].reshape(-1) # Shape (nblobs * npairs * 4,)

    # 3. Create v1 and v2, repeated for nblobs and 4 crossing bounds
    #    First, get the base v1 and v2 from strip_pairs
    pairs_tensor = strip_pairs(nviews)
    base_v1 = pairs_tensor[:, 0] # Shape (npairs,)
    base_v2 = pairs_tensor[:, 1] # Shape (npairs,)

    #    Repeat these for each blob and each of the 4 crossing bounds
    #    We need a tensor that looks like:
    #    [v1_pair0, v1_pair0, v1_pair0, v1_pair0, v1_pair1, v1_pair1, ..., v1_pairN, v1_pairN, v1_pairN, v1_pairN]
    #    repeated nblobs times.

    # Option 1: Using unsqueeze and expand
    # (npairs,) -> (1, npairs, 1, 1) -> (nblobs, npairs, 4, 1)
    v1 = base_v1.unsqueeze(0).unsqueeze(2).unsqueeze(2).expand(nblobs, -1, 4, -1).reshape(-1)
    v2 = base_v2.unsqueeze(0).unsqueeze(2).unsqueeze(2).expand(nblobs, -1, 4, -1).reshape(-1)

    # Each has shape (nblobs * npairs * 4).  
    return v1, r1, v2, r2


def blob_crossings_bounds_new_view(coords, crossings, nviews):
    '''
    This gives the blob bounds in the new view considering all crossings
    including those not "inside" the blob.  See blob_crossings_new_view()
    '''
    nblobs = crossings.shape[0]
    v1, r1, v2, r2 = flatten_crossings(crossings, nviews)
    v3 = torch.full_like(v1, fill_value=nviews)
    pitches = coords.pitch_location((v1,r1), (v2,r2), v3)
    blob_pitches = pitches.reshape(nblobs, -1)
    pmin = torch.min(blob_pitches, dim=1).values
    pmax = torch.max(blob_pitches, dim=1).values
    next_layer = nviews
    lo = coords.pitch_index(pmin, next_layer)
    hi = coords.pitch_index(pmax, next_layer) + 1
    return (lo, hi)


def blob_bounds(coords, crossings, nviews, insides):
    """
    Calculates the half-open ray index range (lo, hi) for each blob in the new view,
    considering only crossings where 'insides' is True.

    Args:
        coords: The coordinate transformation object.
        crossings: The (nblobs, npairs, 4, 2) tensor reflecting the *current* blobs and views.
        nviews: The total number of views that *currently* exist in 'crossings'.
                This will also be the index of the newly added view (the 'next_layer').
        insides: The (nblobs, npairs, 4) boolean tensor indicating which crossings are "inside".

    Returns:
        A tuple (lo, hi) where lo and hi are 1D tensors of shape (nblobs,)
        representing the half-open range [lo, hi) for each blob in the new view.
        If a blob has no 'inside' crossings, its range will be [0, 0).
    """
    nblobs = crossings.shape[0]
    # npairs can be derived from crossings.shape[1]
    # num_crossing_types is always 4, as defined by strip_pair_edge_indices
    num_crossing_types = crossings.shape[2]

    # 1. Flatten crossings to get v1, r1, v2, r2 for pitch_location
    # The 'nviews' passed here is the current total number of views that built the 'crossings' tensor.
    v1, r1, v2, r2 = flatten_crossings(crossings, nviews)

    # 2. Determine v3 (the index of the new view)
    # v3 is constant for all pitches calculated in this function.
    # It represents the index of the 'next_layer' that this function is calculating bounds for.
    v3 = torch.full_like(v1, fill_value=nviews, dtype=torch.long)

    # 3. Calculate pitches for all crossings
    # This returns a 1D tensor of all pitches: (nblobs * npairs * 4,)
    pitches = coords.pitch_location((v1, r1), (v2, r2), v3)

    # 4. Reshape pitches to (nblobs, npairs * 4) to align with flattened 'insides'
    blob_pitches = pitches.reshape(nblobs, -1) # Using -1 infers npairs * 4

    # 5. Flatten 'insides' to (nblobs, npairs * 4) to match blob_pitches
    insides_flat = insides.reshape(nblobs, -1) # Using -1 infers npairs * 4

    # 6. Apply the 'insides' mask to blob_pitches for min/max calculation
    # We use torch.where to conditionally replace values:
    # - For minimum: Replace pitches corresponding to False 'insides' with +infinity.
    #   Any finite valid pitch will be smaller than infinity.
    masked_pitches_for_min = torch.where(
        insides_flat,
        blob_pitches,
        torch.tensor(float('inf'), device=insides.device)
    )

    # - For maximum: Replace pitches corresponding to False 'insides' with -infinity.
    #   Any finite valid pitch will be larger than negative infinity.
    masked_pitches_for_max = torch.where(
        insides_flat,
        blob_pitches,
        torch.tensor(float('-inf'), device=insides.device)
    )

    # 7. Calculate pmin and pmax for each blob across the (npairs * 4) dimension
    pmin = torch.min(masked_pitches_for_min, dim=1).values
    pmax = torch.max(masked_pitches_for_max, dim=1).values

    # 8. Handle cases where a blob has NO 'inside' crossings
    # If all 'insides_flat' values for a specific blob are False, then pmin will be +inf
    # and pmax will be -inf. We need to set these to a sensible default, like [0, 0),
    # to indicate an empty or invalid range for that blob in the new view.
    no_valid_crossings_mask = ~torch.any(insides_flat, dim=1) # True for blobs where ALL insides are False

    # For blobs with no valid crossings, set pmin to 0.0
    pmin = torch.where(no_valid_crossings_mask, torch.tensor(0.0, device=pmin.device), pmin)
    # For blobs with no valid crossings, set pmax to 0.0
    pmax = torch.where(no_valid_crossings_mask, torch.tensor(0.0, device=pmax.device), pmax)

    # 9. Convert pitches (pmin, pmax) to ray indices using coords.pitch_index
    next_layer = nviews # The index of the new view

    lo = coords.pitch_index(pmin, next_layer)
    hi = coords.pitch_index(pmax, next_layer) + 1 # +1 for half-open range

    return (lo, hi)


def bounds_clamp(lo, hi, nmeasures):
    '''
    Clamp per-blob bounds to be consistent with an activity of length nmeasures.

    This assures no values outside of [0,nmeasures] inclusive.

    '''
    lo[lo<0] = 0
    lo[lo>nmeasures] = nmeasures
    hi[hi<0] = 0
    hi[hi>nmeasures] = nmeasures
    return (lo, hi)

def crossing_in_other(coords, v1, r1, v2, r2, v3, rbegin3, rend3, nudge=1e-3):
    '''
    Return True if the crossing of (v1,r1) and (v2,r2) are inside the
    inclusive range [rbegin3,rend3] in view v3.

    These may be 1D tensors.
    '''
    pitches = coords.pitch_location((v1,r1), (v2,r2), v3)

    # find indices after we "nudge" the pitch a little high for the low
    # comparison and a little low for the high comparison in order that we
    # "attract" crossing points just outside of the tested strip.  Note, the
    # pitch_index() does a floor() internally so we test half-open range.
    pinds_lo = coords.pitch_index(pitches+nudge, v3)
    pinds_hi = coords.pitch_index(pitches-nudge, v3)

    pinds = coords.pitch_index(pitches, v3)
    in_other = (pinds >= rbegin3) & (pinds < rend3)

    return in_other


def get_true_runs(activity: torch.Tensor) -> torch.Tensor:
    """
    Finds all consecutive regions of True values in a boolean tensor
    and returns their half-open ranges.

    Args:
        activity: A 1D boolean torch.Tensor.

    Returns:
        A 2D torch.Tensor of shape (N_runs, 2), where N_runs is the number
        of consecutive True segments. Each row is [start_index, end_index_half_open].
        Returns an empty tensor if no True runs are found or activity is empty.
    """
    if activity.numel() == 0:
        return torch.empty(0, 2, dtype=torch.long, device=activity.device)

    # Convert boolean to int (True=1, False=0)
    activity_int = activity.int()

    # Pad with zeros at both ends to ensure all runs are 'closed' by zeros.
    # This simplifies boundary handling.
    padded_activity_int = torch.cat((
        torch.tensor([0], dtype=torch.int, device=activity.device),
        activity_int,
        torch.tensor([0], dtype=torch.int, device=activity.device)
    ))

    # Calculate the difference to find transitions (0->1 for starts, 1->0 for ends)
    diff = padded_activity_int.diff()

    # Find start indices of True runs: where diff changes from 0 to 1
    # We subtract 1 from the padded index to get the original activity index.
    start_indices = (diff == 1).nonzero(as_tuple=True)[0]

    # Find end indices (exclusive) of True runs: where diff changes from 1 to 0
    # We subtract 1 from the padded index to get the original activity index.
    # This index now represents the first element *after* the True run,
    # making the range half-open [start, end_half_open).
    end_indices_half_open = (diff == -1).nonzero(as_tuple=True)[0]

    # If no True runs are found, return an empty tensor
    if start_indices.numel() == 0:
        return torch.empty(0, 2, dtype=torch.long, device=activity.device)

    return torch.stack((start_indices, end_indices_half_open), dim=-1)


def expand_blobs_with_activity(
    blobs: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    activity: torch.Tensor
) -> torch.Tensor:
    """
    Expands existing blobs into new blobs based on intersections with
    consecutive True regions in the new 'activity' tensor.

    Args:
        blobs: The existing (nblobs_old, nviews_old, 2) tensor.
        lo: A (nblobs_old,) tensor of low bounds for the new view, per blob.
        hi: A (nblobs_old,) tensor of high (half-open) bounds for the new view, per blob.
        activity: A 1D boolean torch.Tensor representing the new view's activity.

    Returns:
        A new blobs tensor of shape (nblobs_new, nviews_new, 2), where
        nblobs_new is variable, and nviews_new = nviews_old + 1.
        Returns an empty tensor if no new blobs are formed.
    """
    nblobs_old, nviews_old, _ = blobs.shape
    nviews_new = nviews_old + 1
    device = blobs.device

    # Step 1: Get all consecutive True runs from the new activity tensor.
    # true_runs_all: (N_true_runs, 2)
    true_runs_all = get_true_runs(activity)

    if true_runs_all.numel() == 0:
        # If there are no True segments in the activity, no new blobs can be formed.
        return torch.empty(0, nviews_new, 2, dtype=blobs.dtype, device=device)

    num_true_runs = true_runs_all.shape[0]

    # Step 2: Calculate intersections for every old blob with every True run.
    # Expand lo/hi for each blob to enable broadcasting against all true_runs.
    # lo_expanded: (nblobs_old, 1) -> (nblobs_old, num_true_runs)
    # hi_expanded: (nblobs_old, 1) -> (nblobs_old, num_true_runs)
    lo_expanded = lo.unsqueeze(1).expand(-1, num_true_runs)
    hi_expanded = hi.unsqueeze(1).expand(-1, num_true_runs)

    # Expand true_runs' bounds to enable broadcasting against all blobs.
    # true_runs_all_lo: (1, num_true_runs) -> (nblobs_old, num_true_runs)
    # true_runs_all_hi: (1, num_true_runs) -> (nblobs_old, num_true_runs)
    true_runs_all_lo = true_runs_all[:, 0].unsqueeze(0).expand(nblobs_old, -1)
    true_runs_all_hi = true_runs_all[:, 1].unsqueeze(0).expand(nblobs_old, -1)

    # Calculate the intersection (max of lows, min of highs)
    intersection_lo = torch.max(lo_expanded, true_runs_all_lo)
    intersection_hi = torch.min(hi_expanded, true_runs_all_hi)

    # Determine which intersections are valid (i.e., actually overlap)
    # A valid intersection requires the calculated low bound to be less than the high bound.
    valid_intersections_mask = intersection_lo < intersection_hi

    # Step 3: Identify the (old_blob_index, true_run_index) pairs that form new blobs.
    # These indices will be used to gather the original blob data and the new view data.
    # flat_blob_indices: A 1D tensor where each element is the original blob index
    #                    corresponding to a newly formed blob.
    # flat_true_run_indices: A 1D tensor where each element is the true_run index
    #                        corresponding to a newly formed blob.
    flat_blob_indices, flat_true_run_indices = torch.nonzero(valid_intersections_mask, as_tuple=True)

    total_new_blobs = flat_blob_indices.numel()

    if total_new_blobs == 0:
        # No valid intersections means no new blobs.
        return torch.empty(0, nviews_new, 2, dtype=blobs.dtype, device=device)

    # Step 4: Construct the new_blobs tensor.

    # Gather the original blob data for all newly formed blobs.
    # old_blob_data: (total_new_blobs, nviews_old, 2)
    old_blob_data = blobs[flat_blob_indices]

    # Gather the intersected ranges for the new view.
    # new_view_ranges_flat: (total_new_blobs, 2)
    new_view_ranges_flat = torch.stack((
        intersection_lo[flat_blob_indices, flat_true_run_indices],
        intersection_hi[flat_blob_indices, flat_true_run_indices]
    ), dim=-1)

    # Reshape new_view_ranges to (total_new_blobs, 1, 2) to match the
    # dimensionality of old_blob_data for concatenation along dim=1.
    new_view_ranges_reshaped = new_view_ranges_flat.unsqueeze(1)

    # Concatenate the old blob data with the new view's ranges.
    # This creates the final (nblobs_new, nviews_new, 2) tensor.
    new_blobs = torch.cat((old_blob_data, new_view_ranges_reshaped), dim=1)

    return new_blobs


def apply_activity(coords, blobs, activity, just_blobs=True):
    '''
    Apply activity to blobs to make new blobs with one more view.

    By default, just the new blobs are returned.  If just_blobs is False then
    return tuple of (blobs, crossings, insides) tensors.
    '''
    nviews = blobs.shape[1]

    crossings = blob_crossings(blobs)
    insides = blob_insides(coords, blobs, crossings)
    lo, hi = blob_bounds(coords, crossings, nviews, insides)

    lo, hi = bounds_clamp(lo, hi, activity.shape[0])
    blobs =  expand_blobs_with_activity(blobs, lo, hi, activity)
    if just_blobs:
        return blobs
    return (blobs, crossings, insides)
    
