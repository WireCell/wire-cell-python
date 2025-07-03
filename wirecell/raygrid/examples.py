import math
import torch

def symmetric_views(width=100, height=100, pitch_mag=3,
                    angle=math.radians(60.0)):
    '''
    Return a "views" tensor for 3-plain detector with first two planes
    symmetric about the 3rd.  First two views are horiz/vert bounds.
    '''

    rays = torch.zeros((5, 2, 2, 2))

    # We replicate the original C++ pairs, eliding 1st x-axis
    why = torch.tensor([1.0,0.0])
    zee = torch.tensor([0.0,1.0])

    ll = torch.tensor([0,0])
    lr = torch.tensor([0,width])
    ul = torch.tensor([height,0])
    ur = torch.tensor([height, width])

    # horizontal bounds
    view = 0

    rays[view, 0, 0] = ll
    rays[view, 0, 1] = lr
    rays[view, 1, 0] = ul
    rays[view, 1, 1] = ur

    # vertical bounds
    view = 1

    rays[view, 0, 0] = ll
    rays[view, 0, 1] = ul
    rays[view, 1, 0] = lr
    rays[view, 1, 1] = ur

    # /-wires
    view = 2

    d = torch.tensor([math.cos(angle), math.sin(angle)])
    p = torch.tensor([-d[1], d[0]])

    pjump = 0.5 * pitch_mag * p
    mjump2 = torch.dot(pjump, pjump)
    rays[view, 0, 0] = ul + why * mjump2 / torch.dot(why, pjump)
    rays[view, 0, 1] = ul + zee * mjump2 / torch.dot(zee, pjump)

    pjump = 1.5 * pitch_mag * p
    mjump2 = torch.dot(pjump, pjump)
    rays[view, 1, 0] = ul + why * mjump2 / torch.dot(why, pjump)
    rays[view, 1, 1] = ul + zee * mjump2 / torch.dot(zee, pjump)
    
    # \-wires
    view = 3

    d = torch.tensor([math.cos(angle), -math.sin(angle)])
    p = torch.tensor([-d[1], d[0]])

    pjump = 0.5 * pitch_mag * p
    mjump2 = torch.dot(pjump, pjump)
    rays[view, 0, 0] = ll + why * mjump2 / torch.dot(why, pjump)
    rays[view, 0, 1] = ll + zee * mjump2 / torch.dot(zee, pjump)

    pjump = 1.5 * pitch_mag * p
    mjump2 = torch.dot(pjump, pjump)
    rays[view, 1, 0] = ll + why * mjump2 / torch.dot(why, pjump)
    rays[view, 1, 1] = ll + zee * mjump2 / torch.dot(zee, pjump)
    
    # |-wires
    view = 4

    pw = zee
    pjumpw = pitch_mag * pw

    rays[view, 0, 0] = ll + 0.0 * pjumpw
    rays[view, 0, 1] = ul + 0.0 * pjumpw
    rays[view, 1, 0] = ll + 1.0 * pjumpw
    rays[view, 1, 1] = ul + 1.0 * pjumpw
    
    return rays
