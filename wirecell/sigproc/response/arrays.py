#!/usr/bin/env python3
'''
Deal with responses as numpy arrays
'''

import numpy
from wirecell import units
from .schema import FieldResponse, PlaneResponse, PathResponse


def pr2array(pr, nimperwire = 6, nbinsperwire = 10):
    '''
    Convert a schema.PlaneResponse to a numpy array
    '''
    nwires = len(pr.paths) // nimperwire
    midwire = nwires//2

    nticks = pr.paths[0].current.size
    nimps = int(nwires*nbinsperwire)
    res = numpy.zeros((nimps, nticks))
    pitches = numpy.zeros(nimps)

    for iwire in range(nwires):
        ibin0 = iwire * nimperwire
        for ind in range(nimperwire-1):
            a = pr.paths[ibin0 + ind+0].current
            b = pr.paths[ibin0 + ind+1].current
            m = 0.5 * (a+b)

            p1 = pr.paths[ibin0 + ind+0].pitchpos
            p2 = pr.paths[ibin0 + ind+1].pitchpos
            pm = 0.5*(p1+p2)
            
            obin = iwire * nbinsperwire + ind;

            res[obin] = m
            pitches[obin] = pm

    res = res + numpy.flipud(res)
    pitches = pitches - numpy.flip(pitches)
        
    # for path in pr.paths:
    #     print ("%.3f mm"%(path.pitchpos/units.mm))
    return res,pitches


def toarray(fr):
    '''
    Return FR array representation from FR schema representation.
    '''
    nplanes = len(fr.planes)
    planeid = numpy.zeros(nplanes)

    #print (type(fr.tstart), fr.tstart, type(fr.period), fr.period, fr.speed)
    dat = dict(axis=fr.axis,
               origin=fr.origin,
               tstart=fr.tstart,
               period=fr.period,
               speed=fr.speed,
               locations=numpy.zeros(nplanes),
               pitches=numpy.zeros(nplanes))


    responses = list();
    for iplane, pr in enumerate(fr.planes):
        r,p = pr2array(pr)
        # last_r = r

        #dat['resp%d' % pr.planeid] = r
        responses.append(r)
        dat['bincenters%d' % pr.planeid] = p
        dat['locations'][iplane] = pr.location
        dat['pitches'][iplane] = pr.pitch

    for ind, pr in enumerate(fr.planes):
        dat['resp%d' % pr.planeid] = responses[ind]

    return dat


def toschema(fra):
    '''
    Return FR schema representation from FR array representation
    '''
    planes = []
    for key, val in fra.items():
        if not key.startswith("resp"):
            continue
        plane = int(key[4:])
        bc  = fra[f'bincenters{plane}']
        loc = fra['locations'][plane]
        pit = fra['pitches'][plane]
        
        resp = val
        paths = []
        for row in resp:
            path = PathResponse(row, bc, 0)
            paths.append(path)

        pr = PlaneResponse(paths, plane, loc, pit)
        planes.append(pr)

    return FieldResponse(planes, 
                         axis=fra['axis'],
                         origin=fra['origin'],
                         tstart=fra['tstart'],
                         period=fra['period'],
                         speed=fra['speed']),

def coldelec(fra, gain, shaping):
    '''Return an FR array representation that replaces the response in the given
    FR array represenation with ones that have the cold electronics response
    convolved and with additional fields.
    '''

    from . import electronics

    fra = dict(fra)

    fra["gain"] = gain;
    fra["shaping"] = shaping;

    ncols = 0
    eresp = None
    espec = None
    smeared = list()
    for key, val in fra.items():
        if not key.startswith("resp"):
            continue
        r = val

        if ncols != r.shape[1]:
            ncols = r.shape[1]
            times = [units.ns*(fr.tstart + fr.period * ind) for ind in range(ncols)]

            if eresp is None: # use first response just for its sampling
                eresp = electronics(times, gain, shaping)
                espec = numpy.fft.fft(eresp)

        nrows = r.shape[0]
        #print ("shaping %d x %s" % (nrows, ncols))
        for irow in range(nrows):
            rspec = numpy.fft.fft(r[irow])
            r[irow] = numpy.real(numpy.fft.ifft(rspec*espec))
    fra['eresp'] = eresp
    fra['espec'] = espec

    for ind, pr in enumerate(fr.planes):
        fra['resp%d' % pr.planeid] = responses[ind]

    return fra;


def fr2arrays(fr, gain=None, shaping=None):
    '''
    Return a dict of Numpy arrays.  IF gain and shaping are nonzero,
    convolve with corresponding electronics response.

    Deprecated: use toarray().
    '''
    fra = toarray(fr)

    if gain and shaping:
        return coldelec(fra, gain, shaping)

    return fra


