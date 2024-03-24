#!/usr/bin/env python
'''
Functions to resample FR
'''
import numpy
from wirecell import units
from wirecell.util import lmn
from wirecell.sigproc.response.schema import FieldResponse, PlaneResponse, PathResponse


def resample_one(pr, Ts, Tr, **kwds):
    '''
    Resample the path response pr period Ts to Tr.

    Pass kwds to lmn.interpolate()
    '''
    sig = lmn.Signal(lmn.Sampling(Ts, pr.current.size),
                     wave=numpy.array(pr.current))
    fin = lmn.interpolate(sig, Tr, **kwds)
    pr = PathResponse(fin.wave, pr.pitchpos, pr.wirepos)
    return pr


def resample(fr, Tr, Ts=None, **kwds):
    '''
    Return a resampled version of the fr in schema form.

    If Ts is given, it overrides the period given in the FR.

    Pass kwds to lmn.interpolate().
    '''

    # Beware, fr.period is sometimes imprecise.  See the condition CLI.
    Ts = Ts or fr.period

    planes = []
    for plane in fr.planes:
        paths = []
        for pr in plane.paths:
            pr = resample_one(pr, Ts, Tr, **kwds)
            paths.append(pr)
        planes.append(PlaneResponse(paths, plane.planeid, plane.location,
                                    plane.pitch))
    return FieldResponse(planes, fr.axis, fr.origin, fr.tstart, Tr, fr.speed)


def rolloff(fr, fstart=0.9):
    '''
    Return an FR with a linear roll-off applied.

    The fstart gives frequency to start the roll-off in units of the Nyquist
    frequency.
    '''
    planes = []
    for plane in fr.planes:
        paths = []
        for pr in plane.paths:
            spec = numpy.fft.fft(pr.current)
            N = spec.size
            if N % 2:           # odd
                H = (N-1)//2
                E = 0
            else:
                H = N//2 - 1
                E = 1
            Nrfbeg = round(H * fstart)  # number to start the roll off

            if not Nrfbeg:
                paths.append(pr)
                continue

            Nrf = H-Nrfbeg             # number to roll off
            rf = numpy.hstack([numpy.linspace(1, 0, Nrf),
                               [0]*E, numpy.linspace(0, 1, Nrf)])
            Nrftot = 2*Nrf + E
            spec[1+Nrfbeg: 1+Nrfbeg + Nrftot] *= rf
            wave = numpy.real(numpy.fft.ifft(spec))
            paths.append(PathResponse(wave, pr.pitchpos, pr.wirepos))
        planes.append(PlaneResponse(paths, plane.planeid, plane.location,
                                    plane.pitch))
    return FieldResponse(planes, fr.axis, fr.origin, fr.tstart,
                         fr.period, fr.speed)
