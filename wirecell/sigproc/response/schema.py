#!/usr/bin/env python
'''This module defines an object schema which strives to be generic
enough to describe various sets of field responses including:

    - 1D :: responses which do not extend to inter-pitch regions and
      have no intra-pitch variation.  Responses come from averaged 2D
      field calculations, eg with Garfield.  This is the type
      originally used for LArSoft simulation and deconvoultion for
      some time.

    - 2D :: responses defined on drift paths starting from
      fine-grained points on a line perpendicular to drift and wire
      directions and which spans multiple wire regions.  Responses
      come from 2D field calculations, eg with Garfield.  This is the
      type used in the Wire Cell simulation as developed by Xiaoyue Li
      and Wire Cell deconvolution as developed by Xin Qian.

    - 2.5D :: not supported directly by this schema but 2D responses
      can be made by an average over 2D slices made perpendicular to
      wires/strips in each plane.  These slices may hold 2D field
      calculations or be slices of 3D calculations.  In either case,
      some form of average along the wire/strip direction collapses
      the dimensionality to 2D.

    - 3D :: not supported directly by this schema, but responses
      defined on drift paths starting from fine-grained points on a
      plane perpendicular to nominal drift direction and spanning
      multiple wire regions.  Responses come from 3D field
      calculations, eg with LARF.  Simulation and deconvolution using
      these type of responses are not yet developed.

The schema is defined through a number of `dataclass` types.

Units Notice: any attributes of these classes which are quantities
with units must be in Wire Cell system of units.

Coordinate System Notice: X-axis is along the direction counter to the
nominal electron drift, Y-axis is upward, against gravity, Z-axis
follows from the right-handed cross product of X and Y.  The X-origin
is arbitrary.  In Wire Cell the convention is to take the "location"
of the last (collection) plane (but beware for possible deviations).
A global, transverse origin is not specified but each path response is
at a transverse location given in terms of wire and pitch distances
(positions).  In Wire Cell an origin is set from which these are to be
measured.

'''

import dataclasses
from wirecell.util.codec import dataclass_dictify
from typing import List
import numpy

@dataclasses.dataclass
# @dataclass_dictify
class PathResponse:
    '''A path response.

    This holds the instantaneous induced current along a drift path and the
    position of the start point of this path in wire plane coordinates.

        Note: the path is in wire region: 

        region = int(round(pitchpos/pitch)).

        Note: the path is at the impact position relative to closest
        wire: 

        impact = pitchpos-region*pitch.
    '''

    current: numpy.ndarray | None = None
    '''
    The instantaneous current at steps in time along the path.
    '''
    
    pitchpos: float = 0
    ''' The location of the starting point of the path in pitch (wire plane
    coordinate Z).  '''

    wirepos: float = 0
    ''' The location of the starting point of the path along the wire (wire plane
    coordinate Y, usually 0 for 2D models).  '''
    

@dataclasses.dataclass
# @dataclass_dictify
class PlaneResponse:
    '''A plane response.

    THis holds information about a plane and a set of impulse response waveforms
    at a number of paths.

    '''

    paths: List[PathResponse] | None = None
    '''
    A list of per drift path responses.
    '''

    planeid: int = -1
    '''
    A numerical identifier for the plane.
    '''
    
    location: float = 0
    '''
    Location in the drift direction for this plane.  See FieldResponse.origin.
    '''

    pitch: float = 0
    '''
    The uniform wire pitch used for the path responses of this plane
    '''

@dataclasses.dataclass
# @dataclass_dictify
class FieldResponse:
    '''
    A field response.

    This holds overall scalar info and a set of per plane responses.
    '''


    planes: List[PlaneResponse] | None = None
    '''
    A list of plane responses
    '''

    axis: numpy.ndarray | None = None
    '''
    A 3-array giving the normal, anti-parallel to nominal drift
    '''

    origin: float = 0
    '''
    Distance along axis where drift paths begin.  See PlaneResponse.location.
    Typically 10cm for wires and 20cm for strips+holes.
    '''

    tstart: float = 0
    '''
    Time at which drift paths are considered to begin.
    '''

    period: float = 0
    '''
    The sampling period of the field response.  Typically 100ns for 500ns ADCs.
    '''

    speed: float = 0
    '''
    The average, constant drift speed from origin to collection.  Typically close to 1.6mm/us.
    '''

def asdict(s):
    return {f.name:getattr(s,f.name) for f in dataclasses.fields(s)}
        

    return dataclasses.asdict(s)
