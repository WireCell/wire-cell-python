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

    du = torch.tensor([math.cos(angle), math.sin(angle)])
    pu = torch.tensor([-du[1], du[0]])

    pjumpu = 0.5 * pitch_mag * pu
    mjumpu2 = torch.dot(pjumpu, pjumpu)
    rays[view, 0, 0] = ul + why * mjumpu2 / torch.dot(why, pjumpu)
    rays[view, 0, 1] = ul + zee * mjumpu2 / torch.dot(zee, pjumpu)

    pjumpu = 1.5 * pitch_mag * pu
    mjumpu2 = torch.dot(pjumpu, pjumpu)
    rays[view, 1, 0] = ul + why * mjumpu2 / torch.dot(why, pjumpu)
    rays[view, 1, 1] = ul + zee * mjumpu2 / torch.dot(zee, pjumpu)
    
    # \-wires
    view = 3

    dv = torch.tensor([math.cos(angle), -math.sin(angle)])
    pv = torch.tensor([-du[1], du[0]])

    pjumpv = 0.5 * pitch_mag * pv
    mjumpv2 = torch.dot(pjumpv, pjumpv)
    rays[view, 0, 0] = ll + why * mjumpv2 / torch.dot(why, pjumpv)
    rays[view, 0, 1] = ll + zee * mjumpv2 / torch.dot(zee, pjumpv)

    pjumpv = 1.5 * pitch_mag * pv
    mjumpv2 = torch.dot(pjumpv, pjumpv)
    rays[view, 1, 0] = ll + why * mjumpv2 / torch.dot(why, pjumpv)
    rays[view, 1, 1] = ll + zee * mjumpv2 / torch.dot(zee, pjumpv)
    
    # |-wires
    view = 4

    pw = zee
    pjumpw = pitch_mag * pw

    rays[view, 0, 0] = ll + 0.0 * pjumpw
    rays[view, 0, 1] = ul + 0.0 * pjumpw
    rays[view, 1, 0] = ll + 1.0 * pjumpw
    rays[view, 1, 1] = ul + 1.0 * pjumpw
    
    return rays
