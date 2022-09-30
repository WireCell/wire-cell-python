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
        if type == '3view' or type == 'coldbox' or type == 'protodunevd':
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

import math
def distance(p1, p2):
    return math.sqrt( math.pow((p1[0]-p2[0]),2) 
    + math.pow((p1[1]-p2[1]),2)
    + math.pow((p1[2]-p2[2]),2) )

# merge and sort a list of wires with channel number
# it assumes an input format (channel, wire, begin_xyz, end_xyz)
def merge_wires(wires, plane):
    new_wires = []
    chan2wires = {}
    for chan, wire, beg, end in wires:
        if chan not in chan2wires:
            chan2wires[chan] = [(chan, wire, beg, end)]
        else:
            chan2wires[chan].append((chan, wire, beg, end))
    for i, (ch, seg) in enumerate(chan2wires.items()):
        # if plane==1: print(f'{i} {ch} {seg} {distance(seg[0][2], seg[0][3])}')
        if len(seg) == 1:
            chan, wire, beg, end = seg[0]
            new_wires.append((chan, i, beg, end))
        elif len(seg) == 2:
            p1 = seg[0][2] # begin of a segment
            p2 = seg[0][3] # end of a segment
            p3 = seg[1][2]
            p4 = seg[1][3]
            dists = [ distance(p1,p3), distance(p1,p4), distance(p2,p3), distance(p2,p4) ]
            max_dist = max(dists)
            max_index = dists.index(max_dist)
            if max_index == 0:
                new_wire = (ch, i, p1, p3)
            elif max_index == 1:
                new_wire = (ch, i, p1, p4)
            elif max_index == 2:
                new_wire = (ch, i, p2, p3)
            else:
                new_wire = (ch, i, p2, p4)
            new_wires.append(new_wire)
        else:
            print("Warning: more than two segments for a wire, not expected.")
    return new_wires


def merge_tpc(filename):
    output_filename = f"{filename}-mergetpc.txt"
    tpcplane2wires = {}
    with open(filename) as fp:
        for line in fp.readlines():
            if line.startswith("#"): continue
            line = line.strip()
            if not line: continue
            chunks = line.split()
            chan,tpc,plane,wire = map(int, chunks[:4])
            beg = [float(x) for x in chunks[4:7] ]
            end = [float(x) for x in chunks[7:10]]
            if not (tpc, plane) in tpcplane2wires:
                tpcplane2wires[(tpc, plane)] = [(chan, wire, beg, end)]
            else:
                tpcplane2wires[(tpc, plane)].append((chan, wire, beg, end))
    new_tpcs = [110,120,111,121,112,122,113,123,114,124,115,125,116,126,117,127]
    new_tpcplane2wires = {}
    for new_tpc in new_tpcs:
        # decode the new tpc to real tpc pairs
        # eg, 117 -> 14, 15
        a1 = new_tpc - 100
        d1 = a1%10
        d2 = (a1 - d1)//10
        tpc1 = d1*2
        tpc2 = tpc1 +1
        real_planes = [(tpc1,0), (tpc2,0), (tpc1,1), (tpc2,1)]
        if d2==1:
            real_planes.append((tpc1,2))
        elif d2==2:
            real_planes.append((tpc2,2))
        # merge wires from a real tpc pair
        for plane in [0, 1, 2]:
            tpcplanes = [x for x in real_planes if x[1] == plane]
            wires = []
            for (tpc, plane) in tpcplanes:
                wires.extend(tpcplane2wires[(tpc, plane)])
            new_wires = merge_wires(wires, plane)
            if not (new_tpc, plane) in new_tpcplane2wires:
                new_tpcplane2wires[(new_tpc, plane)] = new_wires
            else:
                new_tpcplane2wires[(new_tpc, plane)].extend(new_wires)

    ofile = open(output_filename, "w")
    ofile.write("# channel       tpc     plane   wire    sx      sy      sz      ex      ey      ez\n")
    for new_tpc, plane in new_tpcplane2wires:
        new_wires = new_tpcplane2wires[(new_tpc, plane)]
        # print(f"anode: {new_tpc} plane: {plane} nwires: {len(new_wires)}")
        for chan, wire, beg, end in new_wires:
            line = f"{chan}\t{new_tpc}\t{plane}\t{wire}"
            for v in beg:
                line += f"\t{v}"
            for v in end:
                line += f"\t{v}"
            ofile.write(line+"\n")
    ofile.close()
    return output_filename