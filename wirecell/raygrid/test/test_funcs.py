import math
import torch
import pytest
from wirecell.raygrid import funcs

def test_vector():
    vec = funcs.vector(torch.tensor([ [0.0,0.0], [1.0,1.0] ]))
    assert torch.all(vec == torch.tensor([1.0,1.0]))

def test_direction():
    vec = funcs.ray_direction(torch.tensor([ [0.0,0.0], [1.0,1.0] ]))
    assert torch.all(vec == torch.tensor([1.0,1.0]) / math.sqrt(2.0))

def test_crossing():
    r1 = torch.tensor( [[0.0, 1.0], [2.0, 1.0]] )
    r2 = torch.tensor( [[0.0, 0.0], [2.0, 2.0]] )
    pt = funcs.crossing(r1, r2)
    assert torch.all(pt == torch.tensor([1.0,1.0]))
    
    with pytest.raises(ValueError):
        funcs.crossing(r1, r1)    
    with pytest.raises(ValueError):
        r0 = torch.tensor( [[0.0, 0.0], [2.0, 0.0]] )
        funcs.crossing(r0, r1)

def test_pitch():
    r0 = torch.tensor( [[0.0, 0.0], [2.0, 0.0]] )
    r1 = torch.tensor( [[0.0, 1.0], [2.0, 1.0]] )    
    p = funcs.pitch(r0, r1)
    assert torch.all(torch.tensor([0.0, 1.0]) == p)
