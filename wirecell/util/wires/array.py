#!/usr/bin/env python
'''
This module processes wires in array form
'''

import numpy
                
def endpoints_from_schema(store, plane=0, face=0, anode=0, detector=0):
    '''
    Return Nx2x3 array of wire endpoints in global coordinate system.

    The wires are taken from the store at the given hiearchy indices.

    '''
    def get_endpoints(wind):
        w = store.wires[wind]
        t = store.points[w.tail]
        h = store.points[w.head]
        return [[t.x,t.y,t.z],[h.x,h.y,h.z]]


    sdet = store.detectors[detector]
    sanode = store.anodes[sdet.anodes[anode]]
    sface = store.faces[sanode.faces[face]]
    splane = store.planes[sface.planes[plane]]
    swires = [get_endpoints(w) for w in splane.wires]

    return numpy.array(swires)


def mean_wire_dir(warr):
    '''
    Return the mean wire direction.
    '''
    # head - tail
    wtot = numpy.sum(warr[:,1] - warr[:,0], axis=0)
    wtot /= numpy.linalg.norm(wtot)
    return wtot


def correct_endpoint_array(warr):

    '''
    Apply corrections for common errors in wires files.

    See wire-cell-toolkit/util/docs/wire-schema.org for schema conventions.
    '''
    wdir = mean_wire_dir(warr)

    def flip_wires():
        warr = numpy.roll(warr, 1, 1)
        wdir *= -1

    # wires should point generally "upward" in +Y
    if 1-abs(wdir[2]) < 1e-4:   # special case of horizontal wires
        if wdir[2] > 0:
            flip_wires();
    elif wdir[1] < 0:
        flip_wires()
        
    eks = numpy.array([1,0,0])

    pdir = numpy.cross(eks, wdir);
    pdir = pdir/numpy.linalg.norm(pdir)

    # order wires in increasing pitch location of their centers.
    order = numpy.dot(0.5*(warr[:,0,:] + warr[:,1,:]), pdir)
    indices = numpy.argsort(order)

    return warr[indices]


def mean_wire_pitch(warr):
    '''
    Return mean wire and pitch displacement vectors.

    Wire array warr must be correct.
    '''
    nwires = warr.shape[0]

    wtot = numpy.sum(warr[:,1] - warr[:,0], axis=0)
    wmag = numpy.linalg.norm(wtot)
    wdir = wtot/wmag
    wmean = wmag / nwires

    eks = numpy.asarray([1.0,0.0,0.0])
    pdir = numpy.cross(eks, wdir);
    pdir = pdir/numpy.linalg.norm(pdir)

    # pick typically mid size wires to avoid magnification of endpoint errors of
    # small wires.
    ind1 = int(0.25*nwires);    
    ind2 = int(0.75*nwires);

    wc1 = 0.5*(warr[ind1][0] + warr[ind1][1])
    wc2 = 0.5*(warr[ind2][0] + warr[ind2][1])
    c2c = wc2 - wc1
    pmean = numpy.dot(pdir, c2c) / (ind2-ind1-1);

    print(f'{wmean=} {pmean=} {pdir=}')
    return (wdir*wmean, pdir*pmean)


def rotation(wire, pitch):
    '''Return the coordinate rotation transform matrix R

    The coordinate system is described as a pair:

        (R, T)

    R is a 3x3 matrix that gives a rotation transformation matrix.  When
    multiplied to a DIRECTION vector (not position vector) expressed in global
    coordinates a DIRECTION vector is formed expressed in wire plane
    coordinates:

        d_w = R @ d_g

    Note, R is orthogonal and so the inverse relation holds:

        d_g = R.T() @ d_w

    See translation()
    '''
    ecks = numpy.array([1,0,0])
    return numpy.vstack((ecks,
                         wire / numpy.linalg.norm(wire),
                         pitch / numpy.linalg.norm(pitch)))

def translation(wires):
    '''
    Return coordinate translation T for wires.

    This returns the center of the "zero wire".  Wires must either be a correct
    Nx2x3 wire array or a 2x3 pair of endpoints.

    The returned vector T is a 3-vector expressed in global coordinates and
    represents the displacement of the wire plane origin from the global origin.
    With R and T one may transform a POSITION expressed in global coordinates to
    a POSITION expressed in wire plane coordinates with:

        p_w = R @ (p_g - T)
    '''
    if len(wires.shape) == 3:
        wires = wires[0]
    return 0.5*(wires[0] + wires[1])
