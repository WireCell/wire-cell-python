import math
import torch


def symmetric_views(width=100., height=100., pitch_mag=3,
                    angle=math.radians(60.0)):
    '''
    Return tensor suitable for giving to Coordinates().  Two wire directions
    are symmetric about the 3rd with givne angle.  First two views are
    horiz/vert bounds.

    This function is all in 2D with a (x,y) coordinate system.  If we consider
    looking in the direction of positive Z-axis then wire direction cross pitch
    direction is Z direction.  The layer directions:

    - pitch points toward U, wire points toward L (horiz bounds)
    - pitch points toward R, wire points toward U (vert bounds)
    - pitch points toward UL, wire points toward LL ("U" plane)
    - pitch points toward UR, wire points toward UL ("V" plane)
    - pitch points toward R, wire points toward U ("W" plane)

    Note, this convention is not directly WCT's usual which puts drift on X axis
    and may swap U/V depending on detector.
    '''
    pitches = torch.zeros((5, 2, 2))


    # horizontal ray bounds, pitch is vertical
    pitches[0] = torch.tensor([[width/2.0, 0], [width/2.0, height]])

    # vertical ray bounds, pitch is horizontal
    pitches[1] = torch.tensor([[0, height/2.0], [width, height/2.0]])

    
    # corners
    ll = torch.tensor([0,0])
    lr = torch.tensor([0,width])

    # /-wires as seen looking in negative-X direction
    w = torch.tensor([math.cos(angle), -math.sin(angle)])
    p = torch.tensor([-w[1], w[0]])
    ## inverse: w = [p[1], -p[0]]
    pitches[2] = torch.vstack([
        lr + 0.5*pitch_mag*p,
        lr + 1.5*pitch_mag*p
    ])

    # \-wires
    w = torch.tensor([math.cos(angle), +math.sin(angle)])
    p = torch.tensor([-w[1], w[0]])
    pitches[3] = torch.vstack([
        ll + 0.5*pitch_mag*p,
        ll + 1.5*pitch_mag*p
    ])

    # |-wires        
    pitches[4] = torch.tensor([[0, height/2.0], [pitch_mag, height/2.0]])

    return pitches
