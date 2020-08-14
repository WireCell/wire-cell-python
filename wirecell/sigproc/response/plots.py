#!/usr/bin/env python

from wirecell import units

import numpy
import matplotlib.pyplot as plt
import matplotlib.lines as lines


def time_linspace(fr, planeid):
    '''
    Return LOW-SIDE bin edges of time samples in system of units [time]
    '''
    pr = fr.planes[planeid]
    period = round(fr.period)
    ntbins = pr.paths[0].current.size
    tmin = fr.tstart
    tmax = tmin + ntbins*period
    tdelta = (tmax-tmin)/ntbins
    #print ('TBINS:', ntbins, tmin, tmax, tdelta, period)
    return numpy.linspace(tmin, tmax, ntbins, endpoint=False)

def hz_linspace(fr, planeid):
    '''
    Return LOW-SIDE bin edges of frequency samples in units Hz.
    '''

    pr = fr.planes[planeid]
    period_s = round(fr.period)/units.s
    nfbins = pr.paths[0].current.size
    hzmin = 0
    hzmax = 1.0/period_s
    hzdelta = hzmax/nfbins
    #print (f'FBINS: {nfbins} {hzmin}Hz {hzmax/1e6}MHz {hzdelta/1000.0}kHz {period_s*1e6:f}us')
    return numpy.linspace(hzmin, hzmax, nfbins, endpoint=False)


def impact_linspace(fr, planeid):
    '''Return LOW-SIDE bin edges of impact locations assuming 20 bins
    covering a wire region.  This means an impact on a wire region
    edge and an impact on the wire fill 1 bin each.  The remaining 4
    impacts fill 2 bins.  This returns a final HIGH-SIDE bin edge.

    '''
    pr = fr.planes[planeid]
    impacts = [path.pitchpos for path in pr.paths]
    impacts.sort()
    idelta = 0.5*(impacts[1] - impacts[0])
    imax = max(map(abs, impacts))
    nibins_f = 2*imax/idelta
    nibins = int(round(nibins_f))
    assert abs(nibins-nibins_f) < 1e-6;
    #print (f'IBINS: {nibins} +/-{imax:f} {idelta:f} {nibins_f:f}')
    return numpy.linspace(-imax, imax, nibins+1)

def impact_linspace_index(ls, pitchpos):
    '''Return the absolute indices holding the pitchpos and the reflected.'''
    nbins = len(ls)-1           # includes final HIGH-SIDE bin edge.
    d = ls[1]-ls[0]
    ind = int(round((pitchpos - ls[0])/d))
    if ind <0 or ind >= nbins:
        raise IndexError(f'out of range ls[0]:{ls[0]} d:{d} pp:{pitchpos} nbins:{nbins}')

    assert ind%2 == 0           # bad linspace
    if ind%10 == 0:             # wire or half way line
        if ind%20 == 10:        # wire
            ind -= 1            # directly populate just below wire
        return ((ind,), (nbins-1-ind,))
    return ((ind-1, ind), (nbins-1-(ind-1), nbins-1-ind))
    

def time_impact_meshgrid(fr, planeid):
    '''
    Return a meshgrid of (time,impact) corresponding to the plane's response
    '''
    tlin = time_linspace(fr, planeid)
    plin = impact_linspace(fr, planeid)
    return numpy.meshgrid(tlin, plin)

def hz_impact_meshgrid(fr, planeid):
    '''
    Return a meshgrid of (frequency in hz, impact) corresponding to the plane's response
    '''
    flin = hz_linspace(fr, planeid)
    plin = impact_linspace(fr, planeid)
    return numpy.meshgrid(flin, plin)


def check_plane(fr, planeid):
    pr = fr.planes[planeid]    

    period = round(fr.period)
    period_prec = abs((fr.period  - period)/period)
    if period_prec > 1e-4:
        print(f'large error from period: {period_prec}: {fr.period} for plane {planeid}')
        raise ValueError('bad response period for plane %d'% planeid)

    tlin = time_linspace(fr, planeid)
    plin = impact_linspace(fr, planeid)

    pdelta = plin[1] - plin[0]
    tdelta = tlin[1] - tlin[0]

    for path in pr.paths:
        pitchpos = path.pitchpos # impact position

        imps,ref_imps = impact_linspace_index(tlin, pitchpos)

        assert len(imps) and len(ref_imps)

        if path.current.size != tlin.size:
            raise ValueError("wrong size response")

        for tind, cur in enumerate(path.current):
            time = tlin[0] + tind*tdelta
            terr = time - tlin[tind]
            if abs(terr) >= 0.001*tdelta:
                print ('time error big:', tind, time, times[pind, tind], terr, tdelta)
                raise ValueError('time error big')

def get_plane(fr, planeid, reflect):
    times, impacts = time_impact_meshgrid(fr, planeid)
    cur = get_current(fr, planeid, reflect);
    return (times, impacts, cur)

def get_current(fr, planeid, reflect):
    pr = fr.planes[planeid]
    times, impacts = time_impact_meshgrid(fr, planeid)
    ilin = impact_linspace(fr, planeid)

    nimps, ntbins = times.shape

    currents = numpy.zeros_like(times)

    imp_uniq = numpy.unique(numpy.sort(impacts.reshape(impacts.size)))
    imp_delta = imp_uniq[1]-imp_uniq[0]
    imp_min = numpy.min(imp_uniq)

    for path in pr.paths:
        assert path.current.size == ntbins
        pitchpos = path.pitchpos
        #print (f'plane:{planeid} pp:{pitchpos} nimps:{nimps} id:{imp_delta} im:{imp_min}')
        imps,ref_imps = impact_linspace_index(ilin, pitchpos)
        
        inds = list(imps)
        if reflect:
            inds += list(ref_imps)
        for tind, cur in enumerate(path.current):
            for ind in inds:
                assert ind >= 0
                assert ind < len(ilin)
                currents[ind, tind] = cur

    return currents

def lg10(current):
    # "log-10" style for response, see SP1 paper: https://arxiv.org/pdf/1802.08709.pdf
    (tdim, pdim) = current.shape
    c = numpy.zeros([tdim, pdim])
    for tind in range(tdim):
        for pind in range(pdim):
            curr = -1.0*current[tind, pind]
            if curr>1e-5: c[tind, pind] = numpy.log10(curr*1e5)
            elif curr<-1e-5: c[tind, pind] = -numpy.log10(-curr*1e5)
            else: c[tind, pind] = 0
    return c

def plot_planes(fr, filename=None, trange=(0,70), region=None, reflect=True):
    '''
    Plot field response as time vs impact positions.

    >>> import wirecell.sigproc.response.persist as per
    >>> fr = per.load("garfield-1d-3planes-21wires-6impacts.json.bz2")
    >>> plot_planes(fr)

    '''

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8.0, 10.5))

    fig.subplots_adjust(left=0.1, right=1.0, top=0.95, bottom=0.05)

    # vlims = [0.1, 0.1, 0.1] # linear style
    vlims = [3, 3, 3] # "log10" style

    for planeid in range(3):
        vlim = vlims[planeid]
        t, p, c = get_plane(fr, planeid, reflect=reflect)
        ax = axes[planeid]
        # ax.axis([65, 90, -20, 20])

        imps = numpy.unique(numpy.sort(p.reshape(p.size)))
        
        # pr = fr.planes[planeid]
        # pitches = [path.pitchpos for path in pr.paths]
        # pitches.sort()
        # dpitch = pitches[1]-pitches[0]
        # pmax = max(map(abs, pitches))
        ax.set_xlim(*trange)
        ax.set_title('Induced Current %s-plane' % 'UVW'[planeid])
        ax.set_ylabel('Pitch [mm]')
        ax.set_xlabel('Time [us]')
        im = ax.pcolormesh(t/units.us, p/units.mm, lg10(c/units.picoampere),
                           vmin=-vlim, vmax=vlim,
                           cmap='jet') # also try seismic
        fig.colorbar(im, ax=[ax], shrink=0.9, pad=0.05)
        if region is not None:
            if region == 0:
                region = imps[10]-imps[0]
            #                          start, stop, step
            ax.set_yticks(imps[::20])
            ax.grid(color='black', which='major', linestyle='-', linewidth=.05)
            ax.plot((trange[0],trange[1]), ( region,  region), color='white', linewidth=0.05)
            ax.plot((trange[0],trange[1]), (      0,       0), color='white', linewidth=0.05, linestyle='--')
            ax.plot((trange[0],trange[1]), (-region, -region), color='white', linewidth=0.05)

        # for iwire in range(10):
        #     ax.axhline(-iwire*3*units.mm, linewidth=1, color='black')
        #     ax.axhline(+iwire*3*units.mm, linewidth=1, color='black')
        #     ax.axhline(-(iwire+0.5)*3*units.mm, linewidth=1, color='gray',
        #                linestyle='dashed')
        #     ax.axhline(+(iwire+0.5)*3*units.mm, linewidth=1, color='gray',
        #                linestyle='dashed')

    if filename:
        print ("Saving to %s" % filename)
        if filename.endswith(".pdf"):
            print ("warning: saving to PDF takes an awfully long time.  Try PNG.")

        fig.savefig(filename,dpi=300) # 

def plot_specs(fr, filename=None):
    '''
    Plot field response spectrum amplitude as frequency vs impact positions.
    '''

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8.0, 10.5))

    fig.subplots_adjust(left=0.1, right=1.0, top=0.95, bottom=0.05)

    # vlims = [0.1, 0.1, 0.1] # linear style
    vlims = [4, 4, 4] # "log10" style

    for planeid in range(3):
        vlim = vlims[planeid]

        flin = hz_linspace(fr, planeid)
        plin = impact_linspace(fr, planeid)
        f,p = numpy.meshgrid(flin, plin)

        c = get_current(fr, planeid, reflect=True)
        s = numpy.fft.fft(c, axis=1)
        a = numpy.absolute(s)
        ax = axes[planeid]
        ax.set_title('Response spectrum amplitude %s-plane' % 'UVW'[planeid])
        ax.set_ylabel('Pitch [mm]')
        ax.set_xlabel('Frequency [MHz]')
        #im = ax.imshow(a, aspect='auto')
        
        half = len(flin)//10
        pmid = len(plin)//2
        pa = pmid-50
        pb = pmid+50
        #val = numpy.log10(a)
        val = a
        im = ax.pcolormesh(f[pa:pb,:half]/1e6,
                           p[pa:pb,:half]/units.mm,
                           val[pa:pb,:half],
#                           vmin=-vlim, vmax=-1,
                           cmap='jet') # also try seismic
        fig.colorbar(im, ax=[ax], shrink=0.9, pad=0.05)

    if filename:
        print ("Saving to %s" % filename)
        if filename.endswith(".pdf"):
            print ("warning: saving to PDF takes an awfully long time.  Try PNG.")

        fig.savefig(filename)


