#!/usr/bin/env python
'''
This holds routines to load "multitpc" format wire geometry files for ICARUS.
'''

from . import schema
from wirecell import units

import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from collections import defaultdict, namedtuple

def load(filename):
    '''
    ADD DESCRIPTION
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
            beg = [float(x)*units.cm for x in chunks[4:7] ]
            end = [float(x)*units.cm for x in chunks[7:10]]
            if beg[1] > end[1]:
                beg,end = end,beg # always point upward in Y

            face = 0            # In ICARUS there is only one face.
            apa = tpc           # Apa and tpc are coceptually identical
            wid = wire          # kept the same from multitpc.py script

            wpid = schema.wire_plane_id(plane, face, apa)

            begind = store.make("point", *beg)
            endind = store.make("point", *end)

            wireind = store.make("wire", wid, chan, seg, begind, endind)
            wpids[wpid].append(wireind)

    def wire_pos(ind):
        '''
        Return a number on which to sort wires. A y-z coordinate combination is
        returned.
        '''
        wire = store.get("wire", ind)
        p1 = store.get("point", wire.tail)
        p2 = store.get("point", wire.head)
        length = ( (p1.z - p2.z)**2 + (p1.y - p2.y)**2 )**0.5
        yz_baricenter = 0.5*(p1.z + p2.z) + (0.5/length)*(p2.y + p1.y)
        return yz_baricenter

    # make and collect planes
    by_apa_face = defaultdict(list)
    for wpid, wire_list in sorted(wpids.items()):
        plane,face,apa = schema.plane_face_apa(wpid)
        wire_list.sort(key = wire_pos)

        #apply some conditions to reverse the wire ordering
        if apa%2 == 0 and plane==1:
            print ("Reversing wire order for p%d f%d a%d" %(plane,face,apa))
            wire_list.reverse()
        elif apa%2 > 0 and plane in [0,1]:
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
