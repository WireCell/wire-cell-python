#!/usr/bin/env python
'''
This module defines an schema of object which describe the
wire-related geometry for the Wire Cell Toolkit.

It covers a hierarchy of object types:

    - Detector

    - Anode (eg, one APA)

    - Face

    - Plane

    - Wire

    - Point

Except for Point, all objects are organizational and describe
parentage.  Points are used to describe the end points of wire
segments (wires) and are MUST be expressed in GLOBAL coordinates.

In the following cases the children are referenced by their parent in
a strictly ordered list:

    - A face orders its child planes following how charge drifts.
      That is, in U/V/W order.

    - A plane orders its child wires in increasing pitch direction
      such that the cross product of the wire direction (see next) and
      this pitch direction gives a direction which is normal to the
      wire plane and anti-parallel to the nominal drift direction for
      that plane.

    - Wire places its head end-point closer to the input of the
      detector electronics.  The wire direction is taken from tail to
      head end-points.  For top DUNE APAs and all other current
      detectors this direction points generally upward and for bottom
      DUNE APAs it points generally downward.

Ordering of all other lists is not defined.

Most objects in the schema have an "ident" attribute.  This number is
made available to user C++ code but is treated as OPAQUE but UNIQUE by
the toolkit.  If an object is referenced in multiple contexts their
ident MUST match.  For example, a wire plane and a field reponse plane
must be correlated by their ident.

Schema objects are held in flat, per type store and a reference to an
object of a particular type is made via its index into the store for
that type.

All values MUST be in the WCT system of units.  Code that generates
these objects may use the Python module wirecell.units to assure this.

Although not specified here, there are places where "front" and "back"
qualifiers are applied to "face".  A "front" face is one for which the
cross product of wire and pitch directions (as described above) is
parallel to the global X-axis.
'''


from collections import namedtuple


class Point(namedtuple("Point", "x y z")):
    '''
    A position.

    :param float x:
    :param float y:
    :param float z:
    '''
    __slots__ = ()


class Wire(namedtuple("Wire","ident channel segment tail head")):
    '''
    A Wire object holds information about one physical wire segment.

    A wire is a ray which points in the direction that signals flow
    toward the electronics.

    :param int ident:
    :param int channel: numerical identifier unique to this conductor.
        It is made available via IWire::channel().  This number is
        treated as opaque by the toolkit except that it is expected to
        uniquely identify an electronics input channel.
    :param int segment: count the number of other wires between this
        wire and the channel input.
    :param int tail: index referencing the tail end point of the wire
        in the coordinate system for the anode face.
    :param int head: index referencing the head end point of the wire
        in the coordinate system for the anode face.
    '''
    __slots__ = ()


class Plane(namedtuple("Plane", "ident wires")):
    '''
    A Plane object collects the coplanar wires.

    :param int ident:
    :param list wires: list of indices referencing the wires that make
        up this plane.  This list MUST be sorted in "increasing wire
        pitch location" as described above.
    '''
    __slots__ = ()


class Face(namedtuple("Face", "ident planes")):
    '''
    A Face collects the wire and conductor planes making up one face
    of an anode plane.

    :param int ident:
    :param list planes: list of indices referencing planes.  This list
        MUST be in sorted in U/V/W order as described above.
    '''
    __slots__ = ()



class Anode(namedtuple("Anode","ident faces")):
    '''
    An Anode object collects together Faces.

    A detector like MicroBooNE has just one face per its single anode.
    protoDUNE/SP has two faces per each of its six anodes (aka APAs).

    :param int ident:
    :param list faces: list indices referencing faces.
    '''
    __slots__ = ()    


class Detector(namedtuple("Detector", "ident anodes")):
    '''
    A detector of anodes.

    This allows one or more sets ot anodes to be defined.  
    :param int ident:
    :param list anodes: list of anodes
    '''
    __slots__ = ()


class Store(namedtuple("Store","anodes faces planes wires points")):
    '''
    A store of collections of the objects of this schema.

    This somewhat awkward indirection of a reference into a store is
    so that multiple things may refer to something without having to
    invent some kind of "pointer" which must be carried by each
    object.

    :param list detectors: list of the Detector objects.
    :param list anodes: list of the Anode objects.
    :param list faces: list of the Face objects
    :param list planes: list of the Plane objects.
    :param list wires: list of the Wire objects.
    :param list points: list of the Point objects.
    '''
    __slots__ = ()

    def __repr__(self):
        return "<Store: %d detectors, %d anodes, %d faces, %d planes, %d wires, %d points>" % \
            (len(self.detectors), len(self.anodes), len(self.faces), len(self.planes), len(self.wires), len(self.points))

def classes():
    import sys, inspect
    ret = list()
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            ret.append(obj)
    return ret

def maker():
    '''
    Return a schema instance maker.

    >>> m = maker()
    >>> hid = m.make('Point', x=..., y=..., z=...)
    >>> tid = m.make('Point', x=..., y=..., z=...)
    >>> wid = m.make('Wire', ident=0, channel=0, segment=0 tail=tid, head=hid)

    >>> wire = m.get('Wire', wid)

    >>> store = m.schema()
    '''
    import sys, inspect
    class SchemaMaker(object):

        def __init__(self):
            self._makers = dict()
            for klass in classes():
                lname = klass.__name__.lower()
                self.__dict__[lname+'s'] = list()
                self._makers[lname] = klass

        def make(self, what, *args):
            klass = self._makers[what]
            collection = self.__dict__[what+'s']
            nthings = len(collection)
            thing = klass(*args)
            collection.append(thing)
            return nthings

        def get(self, what, ind):
            collection = self.__dict__[what+'s']
            return collection[ind]

        def wire_ypos(self, ind):
            wire = self.get("wire", ind)
            p1 = self.get("point", wire.tail)
            p2 = self.get("point", wire.head)
            return 0.5*(p1.y + p2.y)
        def wire_zpos(self, ind):
            wire = self.get("wire", ind)
            p1 = self.get("point", wire.tail)
            p2 = self.get("point", wire.head)
            return 0.5*(p1.z + p2.z)

        def schema(self):
            'Return self as a schema.Store'
            return Store(self.anodes, self.faces, self.planes, self.wires, self.points)
    return SchemaMaker()
    
    
layer_mask = 0x7
face_shift = 3
face_mask = 0x1
apa_shift = 4

def wire_plane_id(plane, face, apa):
    'See WireCellIface/WirePlaneId.h'
    return (plane&layer_mask) | (face << face_shift) | (apa << apa_shift)

def plane_face_apa(wpid):
    #return (wpid&layer_mask, (wpid>>face_shift)&face_mask, wpid>>apa_shift)
    return (wpid&layer_mask, (wpid&(1<<face_shift))>>3, wpid>>apa_shift)

