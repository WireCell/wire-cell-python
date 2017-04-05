#!/usr/bin/env python
'''
Objects for describing wires
'''

from ..geo.shapes import Ray, Point3D

from collections import namedtuple

# The general wire object.  beg/end are tail/head of a vector pointing in
# direction signal travels to electronics.  They must be in Wire Cell Toolkit
# system of units (aka mm).  See wirecell.units.
Wire = namedtuple("Wire","plane ch wip seg beg end")

def bounding_box(wires):
    '''
    Return a bounding box exactly containing the given collection of
    Wire objects.
    '''
    mins = list()
    maxs = list()

    for axis in range(3):
        pts = list()
        pts += [w.beg[axis] for w in wires]
        pts += [w.end[axis] for w in wires]
        mins.append(min(pts))
        maxs.append(max(pts))
    return Ray(Point3D(*mins), Point3D(*maxs))
