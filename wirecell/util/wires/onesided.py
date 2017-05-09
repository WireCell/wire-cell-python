#!/usr/bin/env python
'''
This holds MicroBooNE specific routines related to wire geometry.
'''

import schema

import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

    
def wire_plane_id(plane, face, apa):
    'See WireCellIface/WirePlaneId.h'
    layer_mask = 0x7
    face_shift = 3
    apa_shift = 4
    return (plane&layer_mask) | (face << face_shift) | (apa << apa_shift)

def load(filename):
    '''Load a "celltree wire geometry file".

    Return a list of schema.Wire objects.


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

    class MetaStore(object):

        things = "points wires planes faces anodes".split()

        def __init__(self):
            for thing in self.things:
                self.__dict__[thing] = list()
            
        def make(self, what, *args):
            klass = getattr(schema, what.capitalize())
            collection = self.__dict__[what+'s']
            nthings = len(collection)
            thing = klass(*args)
            collection.append(thing)
            return nthings

        def get(self, what, ind):
            collection = self.__dict__[what+'s']
            return collection[ind]

        def schema(self):
            'Return self as a schema.Store'
            return schema.Store(self.anodes, self.faces, self.planes, self.wires, self.points)


        pass
    store = MetaStore()

    # microboone is single-sided, no wrapping
    segment = 0
    face = 0
    apa = 0

    # temporary per-plane lists of wires to allow sorting before tuplizing.
    planes = [list(), list(), list()]

    with open(filename) as fp:
        for line in fp.readlines():
            if line.startswith("#"):
                continue
            line = line.strip()
            if not line:
                continue
            chunks = line.split()
            ch, plane, wip = [int(x) for x in chunks[:3]]
            beg = [float(x) for x in chunks[3:6]]
            end = [float(x) for x in chunks[6:9]]
            for ind in range(3):                  # some zeros are not
                if abs(beg[ind]) < 1e-13:
                    beg[ind] = 0.0
                if abs(end[ind]) < 1e-13:
                    end[ind] = 0.0
            if end[1] < beg[1]:                       # assure proper
                beg,end = end,beg                     # direction

            begind = store.make("point", *beg)
            endind = store.make("point", *end)
            wpid = wire_plane_id(plane, face, apa)
            wireind = store.make("wire", wpid, ch, segment, begind, endind)
            planes[plane].append(wireind)

    def wire_pos(ind):
        wire = store.get("wire", ind)
        p1 = store.get("point", wire.tail)
        p2 = store.get("point", wire.head)
        return 0.5*(p1.z + p2.z)

    wire_plane_indices = list()
    for plane, wire_list in enumerate(planes):
        wire_list.sort(key = wire_pos)
        index = store.make("plane", plane, wire_list)
        wire_plane_indices.append(index)   
    assert(wire_plane_indices == range(3))
    face_index = store.make("face", 0, wire_plane_indices)
    store.make("anode", 0, [face_index])
    return store.schema()
        


        
