#!/usr/bin/env python
'''
Wires and Channels
'''
import schema
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
    Cover a rectangle with a plane of wires starting along the top of
    the given rectangle and starting at given offset from upper-left
    corner of the rectangle and with angle measured from the vertical
    to the wire direction.  Positive angle means the wire starts going
    down-left from the top of the rectangle.  The channel counts the
    attachment point so is unique only as a count across the top of
    the "electronics plane".
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


# this is generated by "chmap.py" part of the protodune-numbers
# technote and using Shanshan's spreadsheet (sha1)
# 37293bf334e9755819b7f2f4b9ffc72211d4b55c  others/shanshan/ProtoDUNE_APA_Wire_Mapping_091917_v3.xlsx
# the array is from chmap.matrixify()
# NOTE: local wire attachement numbers count from 1.
dune_box_map = numpy.array([
       [('u', 19), ('u', 17), ('u', 15), ('u', 13), ('u', 11), ('v', 19),
        ('v', 17), ('v', 15), ('v', 13), ('v', 11), ('w', 23), ('w', 21),
        ('w', 19), ('w', 17), ('w', 15), ('w', 13)],
       [('u', 9), ('u', 7), ('u', 5), ('u', 3), ('u', 1), ('v', 9),
        ('v', 7), ('v', 5), ('v', 3), ('v', 1), ('w', 11), ('w', 9),
        ('w', 7), ('w', 5), ('w', 3), ('w', 1)],
       [('w', 14), ('w', 16), ('w', 18), ('w', 20), ('w', 22), ('w', 24),
        ('v', 12), ('v', 14), ('v', 16), ('v', 18), ('v', 20), ('u', 12),
        ('u', 14), ('u', 16), ('u', 18), ('u', 20)],
       [('w', 2), ('w', 4), ('w', 6), ('w', 8), ('w', 10), ('w', 12),
        ('v', 2), ('v', 4), ('v', 6), ('v', 8), ('v', 10), ('u', 2),
        ('u', 4), ('u', 6), ('u', 8), ('u', 10)],
       [('u', 29), ('u', 27), ('u', 25), ('u', 23), ('u', 21), ('v', 29),
        ('v', 27), ('v', 25), ('v', 23), ('v', 21), ('w', 35), ('w', 33),
        ('w', 31), ('w', 29), ('w', 27), ('w', 25)],
       [('u', 39), ('u', 37), ('u', 35), ('u', 33), ('u', 31), ('v', 39),
        ('v', 37), ('v', 35), ('v', 33), ('v', 31), ('w', 47), ('w', 45),
        ('w', 43), ('w', 41), ('w', 39), ('w', 37)],
       [('w', 26), ('w', 28), ('w', 30), ('w', 32), ('w', 34), ('w', 36),
        ('v', 22), ('v', 24), ('v', 26), ('v', 28), ('v', 30), ('u', 22),
        ('u', 24), ('u', 26), ('u', 28), ('u', 30)],
       [('w', 38), ('w', 40), ('w', 42), ('w', 44), ('w', 46), ('w', 48),
        ('v', 32), ('v', 34), ('v', 36), ('v', 38), ('v', 40), ('u', 32),
        ('u', 34), ('u', 36), ('u', 38), ('u', 40)]], dtype=object)

def dune_box_map_flatten(dfm = dune_box_map):
    '''Flatten an ASIC channel X number matrix to a dictionary keyed by
    (plane letter, local wire attachment number (1-48 or 1-40).  Value
    is a tuple (ichip, ich) with ichip:{1-8} and ich:{1-16}
    '''
    ret = dict()
    for ichip, row in enumerate(dfm):
        for ich, cell in enumerate(row):
            cell = tuple(cell)
            ret[cell] = (ichip+1, ich+1)
    return ret
dune_box_map_flat = dune_box_map_flatten()

# This is from Manhong.  Wib's are numbered 1-5
#
# |     1    |
# |     2    |
# |     3    |
# |     4    |
# |     5    |
#
# Each WIB has 4 "data connectors" (also power, but we ignore them
# here).  They are arranged in a 2x2 array on each WIB:
# 
# |3,1|
# |4,2|
#
# It is not evident from the diagram if this is viewed from inside the
# cryostat or from outside.
#
# Box numbers are 1-10 going left to right looking at the "front" face
# of the APA and then back right-to-left 11-20.
# It's not yet clear how this reflects for bottom APAs in DUNE FD.
#
# Finally, CE box numbers expressed in rows corresponding to a WIB
# data connector number-1 and columns corresponding to a WIB number-1.
dune_wib_map = numpy.array([
        [ 1, 2, 3, 4, 5],
        [ 6, 7, 8, 9,10],
        [11,12,13,14,15],
        [16,17,18,19,20],
])

def dune_wib_map_flatten(dwcm = dune_wib_map):
    '''Flatten the DUNE CE box WIB connection to array indexed by the box
       number - 1.  The value of each element is (wib#,conn#), both
       numbers are counted from 1.
    '''
    ret = [None]*dwcm.size
    for iconn, row in enumerate(dwcm):
        for iwib, boxn in enumerate(row):
            ret[boxn-1] = (iwib+1, iconn+1)
    return numpy.asarray(ret)
dune_wib_map_flat = dune_wib_map_flatten()

def dune_electronics(wan, iplane, side=1, apa=1):
    '''Return electronics numbers related to the wire attachement point
    (wan) for the given plane.  The wan is assumed to be counting from 1.

    These use the DUNE CE box and WIB maps.
    '''
    iwan = wan - 1

    nwan_per_box = (40,40,48)
    nwan = nwan_per_box[iplane]

    chip_per_box = 8
    chan_per_chip = 16

    rel_box = iwan//nwan #  0-9
    wan_in_box = iwan%nwan
    local_wan = wan_in_box + 1

    abs_boxn = rel_box + 1
    if side < 0: abs_boxn += 10

    #chan_in_box = iwan%nwan                   # 0-127
    #chip_in_box = chan_in_box//chip_per_box   # 0-8
    #chan_in_chip = chan_in_box//chan_per_chip # 0-16

    letter = "uvw"[iplane]
    asic, ch = dune_box_map_flat[(letter, local_wan)]
    wib,wibconn = dune_wib_map_flat[abs_boxn-1]
    
    return dict(
        chip=asic,              # starts from 1
        ch=ch,                  # starts from 1
        local_wan = local_wan,
        box = abs_boxn,
        iplane = iplane,
        letter = letter,
        wib = wib,
        wibconn = wibconn,
        )

def dune_wib_channel(apa=None, wib=None, wibconn=None, chip=None, ch=None):
    '''
    Return a unique channel number such as might likely be set by a WIB.

    All numbers must be specified, non-zero (count from 1) and in their correct range.
    '''
    if not all([apa,wib,wibconn,chip,ch]):
        return
    if apa-1 not in range(16):
        return
    if wib-1 not in range(5):
        return
    if wibconn-1 not in range(4):
        return
    if chip-1 not in range(8):
        return
    if ch-1 not in range(16):
        return
    return int("%d%d%d%d%02d" % (apa,wib,wibconn,chip,ch))

def dune_wrapped(params = protodune_params):

    '''
    Generate a schema.store for channels and wires for wrapped APA.
    '''
    rect = Rectangle(params['width'], params['height'])

    store = schema.maker()

    apa = 1

    # front and back planes, temporary per-plane lists of wires to
    # allow sorting before tuplizing.
    fplanes = [list(), list(), list()]
    bplanes = [list(), list(), list()]
    iface = 0

    planex = params['planex']
    mcpp = params['maxchanperplane']

    for iplane in range(3):
        raw_wires =  wrapped_from_top(params['offsets'][iplane], 
                                      params['angles'][iplane],
                                      params['pitches'][iplane],
                                      rect)
        # (along_pitch, side, channel, seg, p1, p2)
        for iwire, raw_wire in enumerate(raw_wires):
            ap, side, wan, seg, p1, p2 = raw_wire

            # front coordinates of wire endpoints
            fx = planex[iplane]
            fz1, fy1 = p1
            fz2, fy2 = p2

            # back coordinates are rotationally symmetric about Y axis
            bx = -fx
            bz1, bz2 = (-fz1, -fz1)
            by1, by2 = ( fy1,  fy2)
        
            fce = dune_electronics(wan, iplane, +1, apa)
            bce = dune_electronics(wan, iplane, -1, apa)

            fch = dune_wib_channel(**fce)
            bch = dune_wib_channel(**bce)

            ... I think I should put all this mess into an SQLite DB and have full relationship.

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
        print letter, len(wires)
        for wire in wires:
            ap, side, channel, seg, p1, p2 = wire
            print wire
            aps.add(ap)
            sides.add(side)
            channels.add(channel)
    print len(aps),len(sides),len(channels)
