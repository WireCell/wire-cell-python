import torch

def threshold_1d(activity, minimum=0.0):
    '''
    Return pairs of indices of a 1D tensor of activity measured across one
    view.  Each pair of indices gives the begin/end bounds of a half-open range
    that has contiguous activity values above the minimum.

    Returns a tensor of shape (2,N) for N ranges.  The [0] element is the start
    indices for the half-open bounds and the [1] element is the end indices.

    Half open means the range extends from and includes the 1st bound up to but
    excluding the 2nd bound.  The half-open range of [1,4] includes 1, 2 and 3
    but not 4.
    '''
    mask = activity > minimum
    mask = torch.cat((torch.tensor([False]), mask, torch.tensor([False])))
    begs = ((mask[1:]  == True) & (mask[:-1] == False)).nonzero().squeze()
    ends = ((mask[:-1] == True) & (mask[1:]  == False)).nonzero().squeze()
    return torch.vstack((begs, ends))

