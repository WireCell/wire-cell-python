import torch

def pitch(r0, r1):
    '''
    Return the perpendicular vector giving the separation between two
    parallel rays.
    '''
    # along ray 0
    rdir = r0[1] - r0[0]
    # transpose to get unit perpendicular
    uperp = torch.tensor([-rdir[1], rdir[0]]) / torch.norm(rdir)
    # connecting vector between points on either ray
    cvec = r1[0]-r0[0]
    # project onto the perpendicualr
    pdist = torch.dot(cvec, uperp)
    return pdist * uperp


def crossing(r0, r1):
    '''
    Return point where two non-parallel rays cross.  None if 
    '''
    p1 = r0[0]
    p2 = r0[1]
    p3 = r1[0]
    p4 = r1[1]

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if torch.isclose(denominator, torch.tensor(0.0)):
        raise ValueError("parallel lines do not cross")

    t_numerator = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    t = t_numerator / denominator

    intersection_point = p1 + t * (p2 - p1)
    return intersection_point    

def ray_direction(ray):
    n = vector(ray);
    d = torch.linalg.norm(n)
    return n / d

def vector(ray):
    '''
    Return the unit vector along ray direction
    '''
    return ray[1] - ray[0]

def vec_direction(v: torch.Tensor) -> torch.Tensor:
    """
    Returns the unit vector of a given 2D vector.
    Handles the case of a zero vector to prevent division by zero.
    """
    norm = torch.linalg.norm(v)
    if norm == 0:
        return torch.zeros_like(v)
    return v / norm

def combo_partition(n: int, k: int) -> torch.Tensor:
    """
    Return (c,nc) where c holds n-choose-k combinations and nc holds the
    numbers not chosen in c.

    Args:
        n (int): The total number of elements (from 0 to n-1).
        k (int): The number of elements chosen in each combination.

    Returns:
        tuple (c,nc)

        c: shape (Ncombos, k)
        nc: shape (Ncombos n-k)

    """
    c = torch.combinations(torch.arange(n), k)
    if c.numel() == 0:
        raise ValueError(f'no combinations for {n}-choose-{k}')

    num_combinations = c.shape[0]
    
    # Create a tensor representing all possible elements (0 to n-1)
    all_elements_range = torch.arange(n, dtype=torch.int64)

    # 2. Create a boolean mask to identify chosen elements
    # Initialize a mask of shape (num_combinations, n) with all False values.
    # This mask will indicate for each combination (row) which of the 'n'
    # possible elements (columns) are present.
    mask = torch.zeros((num_combinations, n), dtype=torch.bool)

    # Mark chosen elements as true
    mask.scatter_(1, c, True)

    # 3. Invert the mask to get the unchosen elements
    unchosen_mask = ~mask

    # 4. Extract the actual unchosen values and reshape
    nonzero_indices = unchosen_mask.nonzero()

    # If all elements are chosen (k=n), then `unchosen_mask` will be all False,
    # and `nonzero_indices` will be empty. Handle this case.
    if nonzero_indices.numel() == 0:
        raise ValueError(f'no unchosen from {n}-choose-{k}')

    # Extract the unchosen values, which are in the second column of `nonzero_indices`.
    nc_values = nonzero_indices[:, 1]

    # Reshape the flattened list of unchosen values back into the desired shape:
    # `(num_combinations, n - k)`.
    nc = nc_values.reshape(num_combinations, n - k)

    return c, nc
