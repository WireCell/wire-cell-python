#!/usr/bin/env python3
'''
Various functions to generate patterns of depos.
'''

import math
import numpy
from numpy.random import uniform, normal as gauss
from wirecell import units

track_info_types = numpy.dtype([('pmin','3float32'), ('pmax','3float32'),
                                ('tmin', 'float32'), ('tmax', 'float32'),
                                ('step','f4'),       ('eper','f4')])

def lines(tracks, sets, p0, p1, time, eperstep, step_size, track_speed):
    '''
    Generate sets of tracks.

    Tracks are uniform in direction and location within box defined by
    point p0 and p1.  Time may be a scalar number, or an array of
    length 1 or two.  If a pair, it defines range for a uniform
    distribution.  Else, it gives a single time for all depos.

    Each line is made of individual points separated by step_size.
    eperstep gives number of electrons (postive number) per point.
    The track_speed determines point separation in time.
    '''

    bb = list(zip(p0, p1))
    pmid = 0.5 * (p0 + p1)


    collect = dict()
    for iset in range(sets):
        last_id = 0

        datas = list()
        infos = list()
        tinfos = numpy.zeros(tracks, dtype=track_info_types)
        for itrack in range(tracks):

            pt = numpy.array([uniform(a,b) for a,b in bb])
            g3 = numpy.array([gauss(0, 1) for i in range(3)])
            mag = math.sqrt(numpy.dot(g3,g3))
            vdir = g3/mag
            
            t0 = (p0 - pt ) / vdir # may have zeros
            t1 = (p1 - pt ) / vdir # may have zeros
            
            a0 = numpy.argmin(numpy.abs(t0))
            a1 = numpy.argmin(numpy.abs(t1))

            # points on either side bb walls
            pmin = pt + t0[a0] * vdir
            pmax = pt + t1[a1] * vdir

            dp = pmax - pmin
            pdist = math.sqrt(numpy.dot(dp, dp))
            nsteps = int(round(pdist / step_size))

            pts = numpy.linspace(pmin,pmax, nsteps+1, endpoint=True)
            
            if isinstance(time, int) or isinstance(time, float):
                time0 = time
            elif len(time) == 1:
                time0 = time[0]
            else:
                time0 = uniform(time[0], time[1])

            timef = nsteps*step_size/track_speed
            times = numpy.linspace(time0, timef, nsteps+1, endpoint=True)

            dt = timef-time0
            #print(f'nsteps:{nsteps}, pdist:{pdist/units.mm:.1f} mm, dt={dt/units.ns:.1f} ns, {eperstep}')

            tinfos["pmin"][itrack] = pmin
            tinfos["pmax"][itrack] = pmax
            tinfos["tmin"][itrack] = time0
            tinfos["tmax"][itrack] = timef         
            tinfos["step"][itrack] = step_size
            tinfos["eper"][itrack] = eperstep

            # in terms of charge, negative is expected
            charges = numpy.zeros(nsteps+1) + -eperstep

            zeros = numpy.zeros(nsteps+1)

            data = numpy.vstack([
                times,
                charges,
                pts.T,
                zeros,
                zeros])

            ids = numpy.arange(last_id, last_id + nsteps + 1)
            last_id = ids[-1]

            info = numpy.vstack([
                ids,
                zeros,
                zeros,
                zeros])

            datas.append(data)
            infos.append(info)

        datas = numpy.vstack([d.T for d in datas])
        infos = numpy.vstack([i.T for i in infos])

        # datas is now as (n,7)

        timeorder = numpy.argsort(datas[:,0])
        datas = datas[timeorder]
        infos = infos[timeorder]

        collect[f'depo_data_{iset}'] = numpy.array(datas, dtype='float32')
        collect[f'depo_info_{iset}'] = numpy.array(infos, dtype='int32')
        collect[f'track_info_{iset}'] = tinfos

    return collect

def sphere(origin, p0, p1,
           radius=100*units.cm,
           eperstep=5000, step_size=1*units.mm):
    '''
    Generate artificial spherical shell patterns of depos.

    The origin, p0 and p1 are 3-arrays.
    '''

    bb = list(zip(p0, p1))
    pmid = 0.5 * (p0 + p1)

    npoints = int(0.3*(radius/step_size)**2)

    #print(f'generating {npoints} points on sphere of radius {radius}mm')
    pts = list()
    for ipt in range(npoints):
        pt = numpy.array([uniform(a,b) for a,b in bb])
        g3 = numpy.array([gauss(0, 1) for i in range(3)])
        mag = math.sqrt(numpy.dot(g3,g3))
        vdir = g3/mag
        pts.append(origin + vdir * radius)
    zeros = numpy.zeros(npoints)
    charges = numpy.zeros(npoints) - eperstep
    points = numpy.vstack(pts)
    data = numpy.array( numpy.vstack([
        zeros,
        charges,
        points.T,
        zeros,
        zeros]).T, order='C', dtype="float32")
    info = numpy.array( numpy.vstack([
        numpy.arange(0, npoints),
        zeros,
        zeros,
        zeros]).T, order='C', dtype="int32")

    # must send out in shape (npts, 7) and (npts, 4)
    assert (data.shape[1] == 7)
    assert (info.shape[1] == 4)
    assert (info[0][3] == 0)

    return dict(depo_data_0=data, depo_info_0=info)

        
