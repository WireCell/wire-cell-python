#!/usr/bin/env python
'''
This holds MicroBooNE specific routines related to wire geometry.
'''

import schema
from wirecell import units

import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from collections import defaultdict

def load(filename):
    '''Load a "celltree wire geometry file".

    Return a list of schema.Store.


    Somewhere, there exists code to dump wires from larsoft in a text
    format such as what made the files found:

    https://github.com/BNLIF/wire-cell-celltree/tree/master/geometry

    The file is line oriented.  Comment lines may begin with "#" and then have columns of:

    - channel: some channel ID

    - plane: plane number (0 == U, 1 == V, 2 == W)

    - wip: wire index in its plane

    - tail: triplet (sx,sy,sz) starting position of the wire in cm

    - head: triplet (ex,ey,ez) ending position of the wire in cm

    Example lines:

        # channel plane wind sx sy sz ex ey ex
        0 0 0 -6.34915e-14 117.153 0.0352608 -6.34287e-14 117.45 0.548658
        1 0 1 -6.34915e-14 116.807 0.0352608 -6.33552e-14 117.45 1.14866
        ...

    Some assumptions made by wire cell in using these files:

    - There is no wire wrapping, all wires have segment=0.

    - The wire index in plane (wip) counts from 0 for each plane, has no holes and
      increases with increasing Z coordinate.

    '''

    store = schema.maker()

    # microboone is single-sided, no wrapping
    segment = 0
    face = 0
    apa = 0

    # temporary per-plane lists of wires to allow sorting before tuplizing.
    #planes = [list(), list(), list()]
    planes = defaultdict(list)
    with open(filename) as fp:
        for line in fp.readlines():
            if line.startswith("#"):
                continue
            line = line.strip()
            if not line:
                continue
            chunks = line.split()
            ch, plane, wip = [int(x) for x in chunks[:3]]
            beg = [float(x)*units.cm for x in chunks[3:6]]
            end = [float(x)*units.cm for x in chunks[6:9]]
            for ind in range(3):                  # some zeros are not
                if abs(beg[ind]) < 1e-13:
                    beg[ind] = 0.0
                if abs(end[ind]) < 1e-13:
                    end[ind] = 0.0
            if end[1] < beg[1]:                       # assure proper
                beg,end = end,beg                     # direction

            begind = store.make("point", *beg)
            endind = store.make("point", *end)
            wpid = schema.wire_plane_id(plane, face, apa)
            wireind = store.make("wire", wip, ch, segment, begind, endind)
            planes[wpid].append(wireind)

    def wire_pos(ind):
        wire = store.get("wire", ind)
        p1 = store.get("point", wire.tail)
        p2 = store.get("point", wire.head)
        return 0.5*(p1.z + p2.z)

    wire_plane_indices = list()
    for plane, wire_list in sorted(planes.items()):
        wire_list.sort(key = wire_pos)
        index = store.make("plane", plane, wire_list)
        wire_plane_indices.append(index)   
    assert(wire_plane_indices == range(3))
    face_index = store.make("face", 0, wire_plane_indices)
    store.make("anode", 0, [face_index])
    return store.schema()
        


