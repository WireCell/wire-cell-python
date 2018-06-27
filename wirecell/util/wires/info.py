#!/usr/bin/env python
'''
Functions to provide information about wires
'''
from wirecell import units

def p2p(p):
    return dict(x=p.x, y=p.y, z=p.z)

def todict(store):
    '''
    Convert the store to a dict-based representation.  This will
    inflate any duplicate references
    '''
    d_anodes = list()
    for anode in store.anodes:
        d_anode = dict(ident = anode.ident, faces = list())
        for iface in anode.faces:
            face = store.faces[iface]
            d_face = dict(ident = face.ident, planes = list())
            for iplane in face.planes:
                plane = store.planes[iplane]
                d_plane = dict(ident = plane.ident, wires = list())

                #print ("anode:%d face:%d plane:%d" % (anode.ident, face.ident, plane.ident))

                for wind in plane.wires:
                    wire = store.wires[wind]
                    d_wire = dict(ident = wire.ident,
                                  channel = wire.channel,
                                  segment = wire.segment,
                                  head = p2p(store.points[wire.head]),
                                  tail = p2p(store.points[wire.tail]))
                    d_plane["wires"].append(d_wire)
                d_face["planes"].append(d_plane)
            d_anode["faces"].append(d_face)
        d_anodes.append(d_anode)

    # fixme: this should support detectors, for now, just assume one
    return [dict(ident=0, anodes=d_anodes)]


class BoundingBox(object):
    def __init__(self):
        self.minp = None
        self.maxp = None

    def __call__(self, p):
        if self.minp is None:
            self.minp = dict(p)
            self.maxp = dict(p)
            return

        for c,v in self.minp.items():
            if p[c] < v:
                self.minp[c] = p[c]

        for c,v in self.maxp.items():
            if p[c] > v:
                self.maxp[c] = p[c]

    def center(self):
        return {c:0.5*(self.minp[c]+self.maxp[c]) for c in "xyz"}


def summary(store):
    '''
    Return a summary data structure about the wire store.
    '''
    lines = list()
    for det in todict(store):
        for anode in det['anodes']:
            for face in anode['faces']:
                bb = BoundingBox()
                for plane in face['planes']:
                    for wire in plane['wires']:
                        bb(wire['head']);
                        bb(wire['tail']);
                lines.append("anode:%d face:%d X=[%.2f,%.2f]mm Y=[%.2f,%.2f]mm Z=[%.2f,%.2f]mm" % \
                             (anode['ident'], face['ident'],
                              bb.minp['x']/units.mm, bb.maxp['x']/units.mm,
                              bb.minp['y']/units.mm, bb.maxp['y']/units.mm,
                              bb.minp['z']/units.mm, bb.maxp['z']/units.mm))
                for plane in face['planes']:
                    lines.append('\t%d: x=%.2fmm dx=%.4fmm' % \
                                 (plane['ident'],
                                  plane['wires'][0]['head']['x']/units.mm,
                                  (plane['wires'][0]['head']['x']-face['planes'][2]['wires'][0]['head']['x'])/units.mm))
    return lines
