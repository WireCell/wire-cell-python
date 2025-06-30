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

def direction(ray):
    d = vector(ray);
    return d / torch.norm(d)

def vector(ray):
    '''
    Return the unit vector along ray direction
    '''
    return ray[1] - ray[0]


