#!/usr/bin/env python
'''
This holds routines to load "multitpc" format wire geometry files.
'''

from . import schema
from wirecell import units

import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from collections import defaultdict, namedtuple

def load(filename):
    '''
    Load a "multitpc" geometry file.

    Return a list of schema.Wire objects.

    Somewhere, there exists code to dump wires from larsoft in a text
    format such as what made the files found in the file

    protodune-wires-larsoft-v1.txt

    from the wire-cell-data package.  It has columns like:

    # chan tpc plane wire sx sy sz ex ey ez

    And, further the order of rows of identical channel number express
    progressively higher segment count.

        - channel: 0-based ID unique to one conductor in the entire
          detector

        - tpc: 0-based count of the "TPC" which is associated with one
          face of one anode.

        - plane: plane number (0 == U, 1 == V, 2 == W) in the face/tpc

        - wip: wire index in its plane

        - tail: triplet (sx,sy,sz) starting position of the wire in cm

        - head: triplet (ex,ey,ez) ending position of the wire in cm

    Example lines::

        # chan tpc plane wire sx sy sz ex ey ez
        0 0 0   0 -368.926 606.670  229.881     -368.926 605.569 230.672
        0 1 0 400 -358.463 285.577  5.68434e-14 -358.463 606.509 230.672
        0 0 0 800 -368.926 286.518 -1.56319e-13 -368.926   7.61  200.467
        1 0 0   1 -368.926 606.670  229.306     -368.926 604.769 230.673
        1 1 0 401 -358.463 284.777  5.68434e-14 -358.463 605.709 230.672
        1 0 0 801 -368.926 285.718  1.7053e-13  -368.926   7.61  199.892
        ...
        800 0 1   0 -368.449 605.639  0.335  -368.449 606.335 0.835
        800 1 1 400 -358.939 605.648  0.335  -358.939 285.648 230.337
        800 0 1 800 -368.449   7.61  30.4896 -368.449 285.656 230.337
        801 0 1   1 -368.449 604.839  0.335  -368.449 606.335 1.40999
        801 1 1 401 -358.939 604.848  0.335  -358.939 284.848 230.337
        801 0 1 801 -368.449   7.61  31.0646 -368.449 284.856 230.337
        ...
        1600 0 2 0 -367.973 7.61 0.57535 -367.973 606 0.57535
        1601 0 2 1 -367.973 7.61 1.05455 -367.973 606 1.05455
        1602 0 2 2 -367.973 7.61 1.53375 -367.973 606 1.53375

    As shown in the example, the first and second set of three lines
    are from each the same "chan" and represent subsequent segments of
    one condutor.

    Also as shown, there is no sanity in segment direction.

    TPC 1 are on the "front" face and TPC 0 are "back" face in the
    sense that the normal to a front face pointing into the drift
    region is parallel to the global X axis.
    '''

    store = schema.maker()

    wpids = defaultdict(list)
    with open(filename) as fp:
        seg = 0
        last_chan = -1
        for line in fp.readlines():
            if line.startswith("#"):
                continue
            line = line.strip()
            if not line:
                continue
            chunks = line.split()
            chan,tpc,plane,wire = map(int, chunks[:4])
            if chan == last_chan:
                seg += 1
            else:
                seg = 0
            last_chan = chan
            beg = [float(x)*units.cm for x in chunks[4:7] ]
            end = [float(x)*units.cm for x in chunks[7:10]]
            if beg[1] > end[1]:
                beg,end = end,beg # always point upward in Y

            face = (1+tpc)%2    # "front" face is 0.
            apa = tpc//2        # pure conjecture
            wid = wire          # this too

            wpid = schema.wire_plane_id(plane, face, apa)

            begind = store.make("point", *beg)
            endind = store.make("point", *end)
            
            wireind = store.make("wire", wid, chan, seg, begind, endind)
            wpids[wpid].append(wireind)

    def wire_pos(ind):
        '''
        Return a number on which to sort wires.  The z-intercept is
        returned.
        '''
        wire = store.get("wire", ind)
        p1 = store.get("point", wire.tail)
        p2 = store.get("point", wire.head)
        z_intercept = p1.z - p1.y * (p2.z - p1.z) / ( p2.y - p1.y ) # this will blow up with ICARUS!
        return z_intercept

    # make and collect planes
    by_apa_face = defaultdict(list)
    for wpid, wire_list in sorted(wpids.items()):
        plane,face,apa = schema.plane_face_apa(wpid)
        wire_list.sort(key = wire_pos)
        if face == 1:           # to satisfy pitch-order and wire(x)pitch cross product
            print ("Reversing wire order for p%d f%d a%d" %(plane,face,apa))
            wire_list.reverse()
        plane_index = store.make("plane", plane, wire_list)
        by_apa_face[(apa,face)].append(plane_index)

    # make and collect faces
    by_apa = defaultdict(list)
    for (apa,face), plane_indices in sorted(by_apa_face.items()):
        face_index = store.make("face", face, plane_indices)
        by_apa[apa].append(face_index)

    for apa, face_indices in sorted(by_apa.items()):
        store.make("anode", apa, face_indices)

    return store.schema()
        


