#!/usr/bin/env python3
'''
Deal with responses as numpy arrays
'''

import numpy
from wirecell import units

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

def fr2arrays(fr, gain=0, shaping=0):
    '''
    Return a dict of Numpy arrays.  IF gain and shaping are nonzero,
    convolve with corresponding electronics response.
    '''
    nplanes = len(fr.planes)
    planeid = numpy.zeros(nplanes)

    #print (type(fr.tstart), fr.tstart, type(fr.period), fr.period, fr.speed)
    dat = dict(origin=fr.origin,
               tstart=fr.tstart,
               period=fr.period,
               speed=fr.speed,
               locations=numpy.zeros(nplanes),
               pitches=numpy.zeros(nplanes))


    responses = list();
    for iplane, pr in enumerate(fr.planes):
        r,p = pr2array(pr)
        last_r = r

        #dat['resp%d' % pr.planeid] = r
        responses.append(r)
        dat['bincenters%d' % pr.planeid] = p
        dat['locations'][iplane] = pr.location
        dat['pitches'][iplane] = pr.pitch
    

    if gain != 0.0 and shaping != 0.0:
        from . import electronics
        
        dat["gain"] = gain;
        dat["shaping"] = shaping;

        ncols = 0
        eresp = None
        espec = None
        smeared = list()
        for r in responses:
            if ncols != r.shape[1]:
                ncols = r.shape[1]
                times = [units.ns*(fr.tstart + fr.period * ind) for ind in range(ncols)]
                eresp = electronics(times, gain, shaping)
                espec = numpy.fft.fft(eresp)

            nrows = r.shape[0]
            #print ("shaping %d x %s" % (nrows, ncols))
            for irow in range(nrows):
                rspec = numpy.fft.fft(r[irow])
                r[irow] = numpy.real(numpy.fft.ifft(rspec*espec))
        dat['eresp'] = eresp
        dat['espec'] = espec
    for ind, pr in enumerate(fr.planes):
        dat['resp%d' % pr.planeid] = responses[ind]
        
    return dat;

