#!/usr/bin/env python
'''
Wires and Channels
'''
from . import schema
from wirecell import units

import math
from collections import namedtuple

import numpy

# Wire = namedtuple("Wire", "index Chan w1 h1 w2 h2")

class Point(object):
    def __init__(self, *coords):
        self._coords = list(coords)

    def __str__(self):
        s = ",".join([str(a) for a in self._coords])
        return "Point(%s)" % s

    def __repr__(self):
        return str(self)

    @property
    def x(self):
        return self[0]
    @x.setter
    def x(self, val):
        self[0] = val

    @property
    def y(self):
        return self[1]
    @y.setter
    def y(self, val):
        self[1] = val

    def __len__(self):
        return len(self._coords)
    def __getitem__(self, key):
        return self._coords[key]
    def __setitem__(self, key, val):
        self._coords[key] = val
    def __iter__(self):
        return self._coords.__iter__()

    def __abs__(self):
        return Point(*[abs(a) for a in self])

    def __sub__(self, other):
        try:
            return Point(*[(a-b) for a,b in zip(self, other)])
        except TypeError:
            return Point(*[(a-other) for a in self])

    def __add__(self, other):
        try:
            return Point(*[(a+b) for a,b in zip(self, other)])
        except TypeError:
            return Point(*[(a+other) for a in self])

    def __mul__(self, other):
        try:
            return Point(*[(a*b) for a,b in zip(self, other)])
        except TypeError:
            return Point(*[(a*other) for a in self])

    def __div__(self, other):
        try:
            return Point(*[(a/b) for a,b in zip(self, other)])
        except TypeError:
            return Point(*[(a/other) for a in self])
    __truediv__ = __div__

    def dot(self, other):
        return sum([a*b for a,b in zip(self, other)])

    @property
    def magnitude(self):
        return math.sqrt(self.dot(self))

    @property
    def unit(self):
        mag = self.magnitude
        return self/mag

class Ray(object):
    def __init__(self, tail, head):
        self.tail = tail
        self.head = head

    def __str__(self):
        return "%s -> %s" % (self.tail, self.head)

    def __repr__(self):
        return str(self)

    @property
    def vector(self):
        return self.head - self.tail

    @property
    def unit(self):
        return self.vector.unit
    

class Rectangle(object):
    def __init__(self, width, height, center = Point(0.0, 0.0)):
        self.width = width
        self.height = height
        self.center = center

    @property
    def ll(self):
        return Point(self.center.x - 0.5*self.width,
                         self.center.y - 0.5*self.height);

    def relative(self, point):
        return point - self.center

    def inside(self, point):
        r = self.relative(point)
        return abs(r.x) <= 0.5*self.width and abs(r.y) <= 0.5*self.height

    def toedge(self, point, direction):
        '''
        Return a vector that takes point along direction to the nearest edge.
        '''
        p1 = self.relative(point)
        d1 = direction.unit
        
        #print "toedge: p1:%s d1:%s" % (p1, d1)

        corn = Point(0.5*self.width, 0.5*self.height)

        xdir = d1.dot((1.0, 0.0))             # cos(theta_x)
        if xdir == 0:
            tx = None
        else:
            xsign = xdir/abs(xdir)
            dx = xsign*corn.x - p1.x
            tx = dx/d1.x

        ydir = d1.dot((0.0, 1.0))             # cos(theta_y) 
        if ydir == 0:
            ty = None
        else:
            ysign = ydir/abs(ydir)
            dy = ysign*corn.y - p1.y
            ty = dy/d1.y


        if ty is None:
            return d1*tx
        if tx is None:
            return d1*ty

        if tx < ty:            # closer to vertical side
            return d1 * tx
        return d1 * ty



def wrap_one(start_ray, rect):
    '''
    Return wire end points by wrapping around a rectangle.
    '''
    p = rect.relative(start_ray.tail)
    d = start_ray.unit
    ret = [p]
    while True:
        #print "loop: p:%s d:%s" %(p,d)
        jump = rect.toedge(p, d)
        p = p + jump
        ret.append(p)
        if p.y <= -0.5*rect.height:
            break
        d.x = -1.0*d.x                    # swap direction
    return ret
        
    




def wrapped_from_top(offset, angle, pitch, rect):
    '''
    Wrap a rectangle with a plane of wires starting along the top of
    the given rectangle and starting at given offset from upper-left
    corner of the rectangle and with angle measured from the vertical
    to the wire direction.  Positive angle means the wire starts going
    down-left from the top of the rectangle.

    Return list of "wires" (wire segments) as tuple:

        - return :: (along_pitch, side, channel, seg, p1, p2)

        - channel :: counts the attachment point at the top of the
          rectangle from left to right starting from 0

        - side :: identify which side the wire is on, (this value is
          redundant with "seg").

        - seg :: the segment number, ie, how many times the wire's
          conductor has wrapped around the rectangle.

        - p1 and p2 :: end points of wire assuming the original
          rectangle is centered on the origin.
    '''

    cang = math.cos(angle)
    sang = math.sin(angle)
    direc = Point(-sang, -cang)
    pitchv = Point(cang, -sang)

    start = Point(-0.5*rect.width + offset, 0.5*rect.height) + rect.center

    step = pitch / cang
    stop = rect.center.x + 0.5*rect.width

    #print -0.5*rect.width, start.x, step, stop

    wires = list()

    channel = 0
    while True:
        points = wrap_one(Ray(start, start+direc), rect)
        side = 1
        for seg, (p1, p2) in enumerate(zip(points[:-1], points[1:])):
            wcenter = (p1+p2)*0.5 - rect.center
            along_pitch = pitchv.dot(wcenter)
            w = (along_pitch, side, channel, seg, p1, p2)
            wires.append(w)
            side *= -1
        start.x += step
        if start.x >= stop:
            break
        channel += 1
    return wires

def wrapped_from_top_oneside(offset, angle, pitch, rect):
    '''
    Wrap a rectangle with a plane of wires starting along the top of
    the given rectangle and starting at given offset from upper-left
    corner of the rectangle and with angle measured from the vertical
    to the wire direction.  Positive angle means the wire starts going
    down-left from the top of the rectangle.

    Return list of "wires" (wire segments) as tuple:

        - return :: (along_pitch, side, channel, seg, p1, p2)

        - channel :: counts the attachment point at the top of the
          rectangle from left to right starting from 0

        - side :: identify which side the wire is on, (this value is
          redundant with "seg").

        - seg :: the segment number, ie, how many times the wire's
          conductor has wrapped around the rectangle.

        - p1 and p2 :: end points of wire assuming the original
          rectangle is centered on the origin.
    '''

    cang = math.cos(angle)
    sang = math.sin(angle)
    direc = Point(-sang, -cang)
    pitchv = Point(cang, -sang)

    start = Point(-0.5*rect.width + offset, 0.5*rect.height) + rect.center

    step = pitch / cang
    stop = rect.center.x + 0.5*rect.width

    #print -0.5*rect.width, start.x, step, stop

    def swapx(p):
        return Point(2*rect.center.x - p.x, p.y)

    wires = list()

    channel = 0
    while True:
        points = wrap_one(Ray(start, start+direc), rect)
        side = 1
        for seg, (p1, p2) in enumerate(zip(points[:-1], points[1:])):
            if side < 0:
                p1 = swapx(p1)
                p2 = swapx(p2)
            wcenter = (p1+p2)*0.5 - rect.center
            along_pitch = pitchv.dot(wcenter)

            # The same wire can serve both both faces if the
            # coordinate system of each face is related by a rotation
            # of the plane along the x axis.  
            w = (along_pitch, side, channel, seg, p1, p2)

            wires.append(w)

            side *= -1          # for next time
        start.x += step
        if start.x >= stop:
            break
        channel += 1
    return wires



# https://www-microboone.fnal.gov/publications/TDRCD3.pdf
microboone_params = dict(
    # drift is 2.5604*units.meter
    width = 10.368*units.meter,                       # in Z
    height = 2.325*units.meter,                       # in Y
    pitches = [3*units.mm, 3*units.mm, 3*units.mm ],
    # guess at left/right ambiguity
    angles = [+60*units.deg, -60*units.deg,  0.0],
    # 
    offsets = [0.0*units.mm, 0.0*units.mm, 0.0*units.mm],
    # fixme: this is surely wrong
    planex = [9*units.mm, 6*units.mm, 3*units.mm],
    maxchanperplane = 3000,
)

protodune_params = dict(
    width = 2295*units.mm,
    height = 5920*units.mm,
    pitches = [4.669*units.mm, 4.669*units.mm, 4.790*units.mm ],
    # guess at left/right ambiguity
    angles = [+35.707*units.deg, -35.707*units.deg,  0.0],
    # guess based on symmetry and above numbers
    offsets = [0.3923*units.mm, 0.3923*units.mm, 0.295*units.mm],
    # fixme: this is surely wrong
    planex = [15*units.mm, 10*units.mm, 5*units.mm],
    maxchanperplane = 1000,
    )


def onesided_wrapped(params = protodune_params):
    '''
    Generate a schema.store of wires for a onesided but wrapped face.
    
    This does not populate electronics.
    '''
    rect = Rectangle(params['width'], params['height'])

    store = schema.maker()

    apa = 0

    # temporary per-plane lists of wires to allow sorting before tuplizing.
    planes = [list(), list(), list()]
    iface = 0

    planex = params['planex']

    mcpp = params['maxchanperplane']

    for iplane in range(3):
        wires =  wrapped_from_top(params['offsets'][iplane], 
                                      params['angles'][iplane],
                                      params['pitches'][iplane],
                                      rect)

        # a common X because each face has its own coordinate system
        x = planex[iplane]

        # (along_pitch, side, channel, seg, p1, p2)
        for iwire,wire in enumerate(wires):
            ap, side, wan, seg, p1, p2 = wire

            z1, y1 = p1
            z2, y2 = p2

            # side is +/-1, if back side, mirror due to wrapping.  Fixme: this
            # is very sensitive to offsets and will likely result in a shift as
            # one goes from a unwrapped wire to its wrapped neighbor.
            z1 *= side
            z2 *= side

            chplane = iplane+1
            if side < 0:
                chplane += 3
            ch = chplane*mcpp + wan

            wid = mcpp*10*(iplane+1) + (iwire+1)

            begind = store.make("point", x, y1, z1)
            endind = store.make("point", x, y2, z2)
            wireind = store.make("wire", wid, ch, seg, begind, endind)
            planes[iplane].append(wireind)
    
    wire_plane_indices = list()
    for iplane, wire_list in enumerate(planes):
        if iplane == 0:
            wire_list.sort(key = lambda w: -1*store.wire_ypos(w))
        elif iplane == 1:
            wire_list.sort(key = store.wire_ypos)
        else:
            wire_list.sort(key = store.wire_zpos)
        wpid = schema.wire_plane_id(iplane, iface, apa)
        index = store.make("plane", wpid, wire_list)
        wire_plane_indices.append(index)   
    face_index = store.make("face", iface, wire_plane_indices)
    store.make("anode", apa, [face_index])
    return store.schema()


def celltree_geometry():
    '''
    Spit out contents of a file like:

    https://github.com/BNLIF/wire-cell-celltree/blob/master/geometry/ChannelWireGeometry_v2.txt

    columns of:
    # channel plane wire sx sy sz ex ey ez
    '''
        

    #Wire = namedtuple("Wire", "index Chan w1 h1 w2 h2")
    # wire: (along_pitch, side, channel, seg, p1, p2)
    aps = set()
    sides = set()
    channels = set()
    for iplane, letter in enumerate("uvw"):
        rect, wires = protodune_plane_one_side(letter)
        print (letter, len(wires))
        for wire in wires:
            ap, side, channel, seg, p1, p2 = wire
            print (wire)
            aps.add(ap)
            sides.add(side)
            channels.add(channel)
    print (len(aps),len(sides),len(channels))
