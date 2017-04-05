#!/usr/bin/env python
'''
Some simple geometry shape helpers.
'''

class Point(object):
    '''
    A N-D Cartesian point.

    It tries to acts as a collection and a number.
    '''

    def __init__(self, *coords):
        self._coords = list(coords)

    def __str__(self):
        s = ",".join([str(a) for a in self._coords])
        return "Point(%s)" % s

    def __repr__(self):
        return str(self)

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

class Point2D(Point):

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

class Point3D(Point2D):

    @property
    def z(self):
        return self[2]
    @z.setter
    def z(self, val):
        self[2] = val

        
    

class Ray(object):
    '''
    A pair of N-D vectors, each represented by an N-D Point
    '''
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
    

class Rectangle2D(object):
    '''
    An 2D rectangle
    '''
    def __init__(self, width, height, center = Point2D(0.0, 0.0)):
        self.width = width
        self.height = height
        self.center = center

    @property
    def ll(self):
        return Point(self.center.x - 0.5*self.width,
                         self.center.y - 0.5*self.height);

    def relative(self, point2d):
        return point2d - self.center

    def inside(self, point2d):
        r = self.relative(point2d)
        return abs(r.x) <= 0.5*self.width and abs(r.y) <= 0.5*self.height

    def toedge(self, point, direction):
        '''
        Return a vector that takes point along direction to the nearest edge.
        '''
        p1 = self.relative(point)
        d1 = direction.unit
        
        #print "toedge: p1:%s d1:%s" % (p1, d1)

        xdir = d1.dot((1.0, 0.0))             # cos(theta_x)
        xsign = xdir/abs(xdir)
        ydir = d1.dot((0.0, 1.0))             # cos(theta_y) 
        ysign = ydir/abs(ydir)

        corn = Point(0.5*self.width, 0.5*self.height)

        dx = xsign*corn.x - p1.x
        dy = ysign*corn.y - p1.y

        tx = dx/d1.x
        ty = dy/d1.y


        if tx < ty:                           # closer to vertical side
            jump = tx
        else:
            jump = ty

        return d1 * jump


