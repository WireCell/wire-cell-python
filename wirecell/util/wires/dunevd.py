#!/usr/bin/env python
'''

'''

from . import schema
from wirecell import units

import numpy
#import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches

from collections import defaultdict, namedtuple

def load(filename, type='3view'):
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
            apa = tpc         # Apa and tpc are coceptually identical
            wid = wire          # kept the same from multitpc.py script
            if type == 'coldbox':
                apa = tpc % 2   # For the coldbox, "apa" refers to a whole CRU made up of two half-CRU objects in the gdml referred here by their "tcp" number 
                jump_wire = chan - 1600*apa # set segment numbers for U wires jumpered across the middle of the CRU. Seg 0 and Seg 1 will correspond to the same offline channel numbers (128, 255) for the Top CRU (apa = 0) and (1728, 1855) for the bottom CRU (apa = 1)
                if jump_wire >= 128 and jump_wire < 256 and tpc >= 2:
                    seg = 1
                else:
                    seg = 0

            wpid = schema.wire_plane_id(plane, face, apa)

            begind = store.make("point", *beg)
            endind = store.make("point", *end)

            wireind = store.make("wire", wid, chan, seg, begind, endind)
            wpids[wpid].append(wireind)

    def wire_pos(ind, plane, type):
        '''
        Return a number on which to sort wires. A y-z coordinate combination is
        returned.
        '''
        wire = store.get("wire", ind)
        p1 = store.get("point", wire.tail)
        p2 = store.get("point", wire.head)
        # length = ( (p1.z - p2.z)**2 + (p1.y - p2.y)**2 )**0.5
        if type == '3view' or type == 'coldbox' :
            return 0.5*(p1.z + p2.z) + 0.5*(p2.y + p1.y)
        elif type == '3view_30deg' :
            if plane == 0:
                return 0.5*(p2.y + p1.y)
            elif plane == 1:
                return -0.5*(p2.y + p1.y)
            elif plane == 2:
                return 0.5*(p1.z + p2.z)
        elif type == '2view' :
            return 0.5*(p1.z + p2.z) + 0.5*(p2.y + p1.y)
        else :
            raise ValueError('type "{}" not implemented!'.format(type))
        

    by_apa_face = defaultdict(list)
    for wpid, wire_list in sorted(wpids.items()):
        plane,face,apa = schema.plane_face_apa(wpid)
        wire_list.sort(key = lambda ind: wire_pos(ind,plane,type))
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
