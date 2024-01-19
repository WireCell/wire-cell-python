#!/usr/bin/env python
'''
Functions to resample FR
'''
import numpy
from wirecell import units
from wirecell.util import lmn
from wirecell.sigproc.response.schema import FieldResponse, PlaneResponse, PathResponse


def resample_one(pr, Ts, Tr, eps=1e-6):
    '''
    Resample the path response pr period Ts to Tr.
    '''
    sig = lmn.Signal(lmn.Sampling(Ts, pr.current.size), wave = numpy.array(pr.current))
    
    rat = lmn.rational(sig, Tr, eps)

    Nr = rat.sampling.duration / Tr
    if abs(Nr - round(Nr)) > eps:
        raise LogicError('rational resize is not rational')
    Nr = round(Nr)

    res = lmn.resample(rat, Nr)

    rez = lmn.resize(res, sig.sampling.duration)

    pr = PathResponse(rez.wave, pr.pitchpos, pr.wirepos)
    return pr


def resample(fr, Tr=None, Ts=None, eps=1e-6):
    '''Return a resampled version of the fr in schema form.

    Either of both of resampled period Tr or size Nr must not be None.  If one
    is None it will be set so that the total signal duration is retained.  If
    both are given the resampled signal will have different duration.

    If Ts is given, it overrides the period given in the FR.

    '''

    Ts = Ts or fr.period        # beware, fr.period is sometimes imprecise 

    planes = []
    for plane in fr.planes:
        paths = []
        for pr in plane.paths:
            pr = resample_one(pr, Ts, Tr, eps=eps)
            paths.append(pr)
        planes.append(PlaneResponse(paths, plane.planeid, plane.location, plane.pitch))
    return FieldResponse(planes, fr.axis, fr.origin, fr.tstart, Tr, fr.speed)
