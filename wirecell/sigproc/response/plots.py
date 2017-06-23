#!/usr/bin/env python

from wirecell import units

import numpy
import matplotlib.pyplot as plt
import matplotlib.lines as lines


def get_plane(fr, planeid, reflect=True):
    pr = fr.planes[planeid]

    period = fr.period

    ntbins = pr.paths[0].current.size
    tmin = fr.tstart
    tmax = tmin + ntbins*period
    tdelta = (tmax-tmin)/ntbins
    print 'TBINS:', ntbins, tmin, tmax, tdelta, period

    pitches = [path.pitchpos for path in pr.paths]
    pdelta = pitches[1] - pitches[0]
    pmax = max(map(abs, pitches)) + 0.5*pdelta   # bin centered
    pmin = -pmax
    npbins = int(round((pmax-pmin)/pdelta))
    print 'PBINS:', npbins, pmin, pmax, pdelta, (pmax-pmin)/pdelta

    tlin = numpy.linspace(tmin, tmax, ntbins+1)
    plin = numpy.linspace(pmin, pmax, npbins+1)

    # print 'T:', tlin
    # print 'P:', plin

    times, pitches = numpy.meshgrid(tlin, plin)
    currents = numpy.zeros((npbins, ntbins))

    for path in pr.paths:
        pitch = path.pitchpos
        pind = int(round((pitch - pmin)/pdelta))
        pind = max(0, pind)
        pind = min(npbins, pind)

        pind_ref = int(round((-pitch - pmin)/pdelta))
        pind_ref = max(0, pind_ref)
        pind_ref = min(npbins, pind_ref)

        assert path.current.size == ntbins
        for tind, cur in enumerate(path.current):

            time = tmin + tind*tdelta
            terr = time - times[pind, tind]

            # sanity check on indexing:
            center_line = pitches[pind, tind] + 0.5*pdelta
            perr = pitch - center_line
            if abs(terr) >= 0.001*tdelta:
                print 'time:', tind, time, times[pind, tind], terr
            if abs(perr) >= 0.001*pdelta:
                print 'pitch:', pind, pitch, pitches[pind, tind], perr
            assert abs(terr) < 0.001*tdelta
            assert abs(perr) < 0.001*pdelta
            if pind >= npbins:
                print 'pitch:', pind, pitch, pitches[pind, tind], perr

            currents[pind, tind] = cur

            if reflect:
                currents[pind_ref, tind] = cur

    return (times, pitches, currents)


def plot_planes(fr, filename=None):
    '''
    Plot field response as time vs impact positions.

    >>> import wirecell.sigproc.response.persist as per
    >>> fr = per.load("garfield-1d-3planes-21wires-6impacts.json.bz2")
    >>> plot_planes(fr)

    '''

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8.0, 10.5))

    fig.subplots_adjust(left=0.05, right=1.0, top=0.95, bottom=0.05)

    vlims = [0.1, 0.1, 0.1]

    for planeid in range(3):
        vlim = vlims[planeid]
        t, p, c = get_plane(fr, planeid)
        print t.shape, p.shape, c.shape
        ax = axes[planeid]
        ax.axis([65, 90, -20, 20])
        ax.set_title('Induced Current %s-plane' % 'UVW'[planeid])
        ax.set_ylabel('Pitch [mm]')
        ax.set_xlabel('Time [us]')
        im = ax.pcolormesh(t/units.us, p/units.mm, c/units.picoampere,
                               vmin=-vlim, vmax=vlim,
                               cmap='seismic')
        fig.colorbar(im, ax=[ax], shrink=0.9, pad=0.05)

        for iwire in range(10):
            ax.axhline(-iwire*3*units.mm, linewidth=1, color='black')
            ax.axhline(+iwire*3*units.mm, linewidth=1, color='black')
            ax.axhline(-(iwire+0.5)*3*units.mm, linewidth=1, color='gray',
                       linestyle='dashed')
            ax.axhline(+(iwire+0.5)*3*units.mm, linewidth=1, color='gray',
                       linestyle='dashed')

    if filename:
        print ("Saving to %s" % filename)
        if filename.endswith(".pdf"):
            print ("warning: saving to PDF takes an awfully long time.  Try PNG.")

        fig.savefig(filename)
