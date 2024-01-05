import numpy as np
from wirecell.util.bbox import *


def test_slice():
    assert slice(0,4) == union_slice(slice(0,4), slice(1,2)) # inside
    assert slice(0,4) == union_slice(slice(1,2), slice(0,4)) # outside
    assert slice(0,4) == union_slice(slice(0,1), slice(2,4)) # gap
    assert slice(0,4) == union_slice(slice(0,4), slice(1,4)) # overlap    

def test_slice():
    assert np.all(np.array([0,1,2,3]) == union_array(slice(0,4), slice(1,2), order="ascending"))
    assert np.all(np.array([1,0,2,3]) == union_array(slice(1,2), slice(0,4), order="seen"))
    assert np.all(np.array([1,0,1,2,3]) == union_array(slice(1,2), slice(0,4), order=None))
    assert np.all(np.array([2,3,0]) == union_array(slice(2,4), slice(0,1), order="seen"))
    assert np.all(np.array([3,2,1,0]) == union_array(slice(0,4), slice(1,4), order="descending"))

def test_bbox():
    bb1 = (slice(0,2), slice(10,12))
    bb2 = (slice(3,5), slice(13,15))
    u1 = union(bb1, bb2, form="slices")
    assert slice(0,5) == u1[0]
    assert slice(10,15) == u1[1]
