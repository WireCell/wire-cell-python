#!/usr/bin/env python
'''
This module defines an object schema which strives to be generic
enough to describe various sets of field responses including:

    - 0D :: responses which do not extend to inter-pitch regions and
      have no intra-pitch variation.  Responses come from averaged 2D
      field calculations, eg with Garfield.  This is the type
      originally used for LArSoft simulation and deconvoultion for
      some time.

    - 1D :: responses defined on drift paths starting from
      fine-grained points on a line perpendicular to drift and wire
      directions and which spans multiple wire regions.  Responses
      come from 2D field calculations, eg with Garfield.  This is the
      type used in the Wire Cell simulation as developed by Xiaoyue Li
      and Wire Cell deconvolution as developed by Xin Qian.

    - 2D :: responses defined on drift paths starting from
      fine-grained points on a plane perpendicular to nominal drift
      direction and spanning multiple wire regions.  Responses come
      from 3D field calculations, eg with LARF.  Simulation and
      deconvolution using these type of responses are not yet
      developed.

The schema is defined through a number of `namedtuple` collections.

Units Notice: any attributes of these classes which are quantities
with units must be in Wire Cell system of units.

Coordinate System Notice: X-axis is along the direction counter to the
nominal electron drift, Y-axis is upward, against gravity, Z-axis
follows from the right-handed cross product of X and Y.  The X-origin
is arbitrary.  In Wire Cell it will be taken from the "location" of
the last (collection) plane.  A global, transverse origin is not
specified but each path response is at a transverse location given in
terms of wire and pitch distances (positions).  In Wire Cell an origin
is set from which these are to be measured.
'''

from collections import namedtuple

class FieldResponse(namedtuple("FieldResponse","planes axis origin tstart period speed")):
    '''
    :param list planes: List of PlaneResponse objects.
    :param list axis: A normalized 3-vector giving direction of axis
        (anti)parallel to nominal drift direction.
    :param float origin: location along the X-axis where drift paths
        begin (see PlaneResponse.location).
    :param float tstart: the time at which drift paths are considered
        to begin.
    :param float period: the sampling period of the field response.
    :param float speed: the nominal drift speed used in the
        calculation.
    '''
    __slots__ = ()




class PlaneResponse(namedtuple("PlaneResponse","paths planeid location pitch")):
    '''
    :param list paths: List of PathResponse objects.
    :param int planeid: A numerical identifier for the plane,
        typically [0,1,2].
    :param float location: Location in drift direction of this plane
        (see FieldResponse.origin).
    :param float pitch: The uniform wire pitch used for the path
        responses of this plane.
    '''
    __slots__ = ()
    

class PathResponse(namedtuple("PathResponse", "current pitchpos wirepos")):
    '''
    :param array current: A numpy array holding the induced current
        for the path on the wire-of-interest.
    :param float pitchpos: The position in the pitch direction to the
        starting point of the path.
    :param float wirepos: The position along the wire direction to the
        starting point of the path.

        Note: the path is in wire region: 

        region = int(round(pitchpos/pitch)).

        Note: the path is at the impact position relative to closest
        wire: 

        impact = pitchpos-region*pitch.
    '''
    __slots__ = ()


