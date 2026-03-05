# here we use firewall other wcpy code that expects numpy
from wirecell.util.wires.array import correct_endpoint_array, mean_wire_pitch, endpoints_from_schema
from wirecell.util.wires.persist import load as load_detector

import torch

def load(detector, apa_idx, correct=True):
    """
    Return tuple of wire arrays (uwires, vwires, wwires).

    Each wire array is shape (Nwires, 2, 3) giving 2 3D endpoints.

    By default wires will be corrected.
    """
    store = load_detector(detector)

    ret = list()

    # warr : (N, 2, 3)
    #        (wire_idx, [start, end], coord_idx)
    for plane_idx in range(3):
        warr = endpoints_from_schema(
            store, plane = plane_idx, anode = apa_idx
        )
        if correct:
            warr = correct_endpoint_array(warr)
        ret.append(torch.from_numpy(warr))
    return tuple(ret)

def to2d(threed):
    x = threed[2]
    y = threed[1]
    return torch.tensor((x,y))

def make_views(uwires, vwires, wwires, width, height):

    views = torch.zeros((5, 2, 2))
    # horizontal ray bounds, pitch is vertical
    views[0] = torch.tensor([[width/2.0, 0], [width/2.0, height]])

    # vertical ray bounds, pitch is horizontal
    views[1] = torch.tensor([[0, height/2.0], [width, height/2.0]])

    for iplane, wires in enumerate((uwires, vwires, wwires)):
        _, pitch = mean_wire_pitch(wires.numpy())
        center = 0.5*(wires[0][0] + wires[0][1])
        pitch = to2d(pitch)
        center = to2d(center)
        views[iplane+2][0] = center
        views[iplane+2][1] = center + pitch
    return views

def to_coordinates(wires):
    """
    Return a Coordinates 
    """
        
