#!/usr/bin/env python
import pytest
from math import sqrt
import numpy
from wirecell.units import degree as deg, radian as rad

from wirecell.gen.linegen import (
    direction_to_tpc_angles,
    tpc_angles_to_direction,
    wp_direction_to_global_direction,
    global_direction_to_wp_direction,
)


# This is the R matrics for PDSP but manually fixing R[2] to be unity matrix.
Rs = numpy.array([
    [[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
     [ 0.00000000e+00,  8.12012227e-01,  5.83640422e-01],
     [ 0.00000000e+00, -5.83640422e-01,  8.12012227e-01]],

    [[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
     [ 0.00000000e+00,  8.12012228e-01, -5.83640422e-01],
     [-0.00000000e+00,  5.83640422e-01,  8.12012228e-01]],

    [[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
     [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00],
     [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]])


def assert_close(a,b,eps=1e-6):
    for aa,bb in zip(a,b):
        assert aa == pytest.approx(bb)

def test_round_trip():
    '''
    From angles to dir to global and back.
    '''

    y_xz = 90*deg, 10*deg

    wp_dir = tpc_angles_to_direction(*y_xz)
    got = direction_to_tpc_angles(wp_dir)
    assert_close(y_xz, got)

    for plane in range(3):
        R = Rs[plane]
        g_dir = wp_direction_to_global_direction(R, wp_dir)
        wp_dir2 = global_direction_to_wp_direction(R, g_dir)
        assert_close(wp_dir, wp_dir2)


def test_wplane_is_global():
    '''
    The W-plane is parallel to global coordinate system.
    '''
    y_xz = 90*deg, 10*deg
    wp_dir = tpc_angles_to_direction(*y_xz)
    got = direction_to_tpc_angles(wp_dir)
    g_dir = wp_direction_to_global_direction(Rs[2], wp_dir)
    assert_close(wp_dir, g_dir)    


def test_select_angles():
    '''
    Select angles give us expected directions.
    '''
    got = tpc_angles_to_direction(90*deg, 45*deg)
    assert_close(got, (sqrt(0.5), 0, sqrt(0.5)))

    got = tpc_angles_to_direction(90*deg, 30*deg)
    assert_close(got, (0.5, 0, 0.866025))

    got = tpc_angles_to_direction(90*deg, 0*deg)
    assert_close(got, (0, 0, 1))


def test_orthogonal_rotation():    
    '''
    The columns of rotation matrics must be mutually orthogonal
    '''
    for plane in range(3):
        R = Rs[plane]
        dots = [numpy.dot(R[:,i], R[:,(i+1)%3]) for i in range(3)]
        assert_close( (0,0,0), dots)
