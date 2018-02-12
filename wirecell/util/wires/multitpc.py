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
        0 0 0 0 -368.926 606.67 229.881 -368.926 605.569 230.672
        0 1 0 400 -358.463 285.577 5.68434e-14 -358.463 606.509 230.672
        0 0 0 800 -368.926 286.518 -1.56319e-13 -368.926 7.61 200.467
        1 0 0 1 -368.926 606.67 229.306 -368.926 604.769 230.673
        1 1 0 401 -358.463 284.777 5.68434e-14 -358.463 605.709 230.672
        1 0 0 801 -368.926 285.718 1.7053e-13 -368.926 7.61 199.892

    As shown in the example, the first and second set of three lines
    are from each the same "chan" and represent subsequent segments of
    one condutor.
    '''

    store = schema.maker()

    planes = defaultdict(list)
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

            face = tpc%2        # checkme: this is 
            apa = tpc//2        # pure conjecture
            wip = wire          # this too

            wpid = schema.wire_plane_id(plane, face, apa)

            begind = store.make("point", *beg)
            endind = store.make("point", *end)
            
            wireind = store.make("wire", wip, chan, seg, begind, endind)
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
    #assert(wire_plane_indices == range(3))
    print ("Got wire_plane_indices:",wire_plane_indices)
    face_index = store.make("face", 0, wire_plane_indices)
    store.make("anode", 0, [face_index])
    return store.schema()
        


