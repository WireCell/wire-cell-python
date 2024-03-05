#!/usr/bin/env python
'''The simple splat / sim+SP (ssss) test is used, in part, to reproduce the
signal biase, efficiency and resolution metric used in the MicroBooNE SP-1
paper.

'''

import dataclasses
import numpy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from wirecell import units
from wirecell.util.peaks import (
    BaselineNoise,
    baseline_noise, 
    gauss as gauss_func
)
import logging
log = logging.getLogger("wirecell.test")


def relbias(a,b):
    '''
    Return (a-b)/a where a is nonzero, return zero o.w..
    '''
    rb = numpy.zeros_like(a)
    ok = b>0
    rb[ok] = a[ok]/b[ok] - 1
    return rb


@dataclasses.dataclass
class Frame:
    '''
    Represent a "frame" loaded from file.
    '''

    filename: str | None
    '''
    Filename from which the frame was taken.
    '''
    frame: numpy.ndarray
    '''
    The frame array
    '''
    extent: tuple
    '''
    Frame extent in time and channel: (t0,tf,cmin,cmax+1)
    '''
    origin: str
    '''
    Origin option for imshow().
    '''
    tick: float
    '''
    Sample period in WCT system of units
    '''

def load_frame(fname, tag="*", ident=0, trange=None, tshift=None):
    '''
    Load a frame with time values in explicit units.

    If trange is given it is a tuple providing (t0,tf) in system of units.

    If tshift is given it is ADDED to the t0 (tickinfo[0]).  
    '''
    fp = numpy.load(fname)
    suffix = f'{tag}_{ident}'
    f = fp["frame_"+suffix]
    t0, tick, tbin = fp["tickinfo_"+suffix]
    if tshift:
        t0 += tshift
    c = fp["channels_"+suffix]

    # assure channels are ordered
    c2 = numpy.array(c)
    numpy.sort(c2)
    assert numpy.all(c == c2)

    cmin = numpy.min(c)
    cmax = numpy.max(c)
    nch = cmax-cmin+1
    ff = numpy.zeros((nch, f.shape[1]), dtype=f.dtype)
    for irow, ch in enumerate(c):
        ff[cmin-ch] = f[irow]
    origin = "lower"            # lower flips putting [0,0] at bottom
    ff = numpy.flip(ff, axis=0)
    array_t0 = t0 + tbin*tick
    array_tf = array_t0 + ff.shape[1]*tick

    if trange:
        dt0 = (trange[0] - array_t0)/tick
        dtf = (trange[1] - array_tf)/tick
        print(f'{dt0=} {dtf=}')
        dt0 = round(dt0)
        dtf = round(dtf)
        if dt0 > 0:
            ff = ff[:, dt0:]
        if dt0 < 0:
            ff = numpy.hstack([numpy.zeros( (ff.shape[0], -dt0) ), ff])
        array_t0 = trange[0]

        if dtf > 0:
            ff = numpy.hstack([ff, numpy.zeros( (ff.shape[0], dtf) )])
        if dtf < 0:
            ff = ff[:, :dtf]
        array_tf = trange[1]
        print(f'{trange=} {dt0=} {dtf=} {ff.shape=}')

    array_extent = (array_t0, array_tf, cmin, cmax+1)

    return Frame(fname, ff, array_extent, origin, tick)

def plot_frame(gs, fr, channel_ranges=None, which="splat", tit=""):
    '''
    Plot one Frame as 2D and 2x1D projections.
    '''
    tunits = units.us

    t0,tf,c0,cf = fr.extent
    t0_us = t0/tunits
    tf_us = tf/tunits
    extent_us = (t0_us, tf_us, c0, cf)

    gs = GridSpecFromSubplotSpec(2,2, subplot_spec=gs,
                  height_ratios = [5,1], width_ratios = [6,1])                                 

    # 2D chan vs time frame
    fax = plt.subplot(gs[0,0])
    # 1D time projection
    tax = plt.subplot(gs[1,0], sharex=fax)
    # 1D channel projection
    cax = plt.subplot(gs[0,1], sharey=fax)

    cax.set_xlabel(which)
    fax.set_ylabel("channel")
    if which=="signal":
        tax.set_xlabel("time [us]")

    if tit:
        plt.title(tit)
    plt.setp(fax.get_xticklabels(), visible=False)
    plt.setp(cax.get_yticklabels(), visible=False)
    if which=="splat":
        plt.setp(tax.get_xticklabels(), visible=False)

    im = fax.imshow(fr.frame, extent=extent_us, origin=fr.origin,
                    aspect='auto', vmax=500, cmap='hot_r')

    tval = fr.frame.sum(axis=0)
    t = numpy.linspace(t0_us, tf_us, fr.frame.shape[1]+1,endpoint=True)
    tax.plot(t[:-1], tval)      # all channels
    if channel_ranges:
        for p, chans in zip("UVW",channel_ranges): # fixme: map to plane labels is only an assumption!
            val = fr.frame[chans,:].sum(axis=0)
            c1 = chans.start
            c2 = chans.stop
            tax.plot(t[:-1], val, label=p)
            fax.plot([t0_us,tf_us], [c1,c1])
            fax.text(t0_us + 0.1*(tf_us-t0_us), c1 + 0.5*(c2-c1), p)
        fax.plot([t0_us,tf_us], [c2-1,c2-1])
        tax.legend()
    
    cval = fr.frame.sum(axis=1)
    c = numpy.linspace(fr.extent[2],fr.extent[3],fr.frame.shape[0]+1,endpoint=True)
    cax.plot(cval, c[:-1])

    return im

def plot_frames(spl, sig, channel_ranges, title=""):
    '''Plot the two Frame objects spl (splat) and sig (signal).

    Channel ranges gives list of pair of channel min/max to interpret as
    contiguous rows on the Frame.array.

    '''
    fig = plt.figure()
    pgs = GridSpec(1,2, figure=fig, width_ratios = [7,0.2])
    gs = GridSpecFromSubplotSpec(2, 1, pgs[0,0])
    im1 = plot_frame(gs[0], spl, channel_ranges, which="splat")
    im2 = plot_frame(gs[1], sig, channel_ranges, which="signal")
    fig.colorbar(im2, cax=plt.subplot(pgs[0,1]))
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    
def plot_plane(spl_act, sig_act, nsigma=3.0, title=""):
    '''
    Plot splat and signal activity for one plane.

    '''
    # bias of first w.r.t. second
    bias1 = relbias(sig_act, spl_act)
    bias2 = relbias(spl_act, sig_act)

    plt.clf()
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    if title:
        plt.suptitle(title)
    args=dict(aspect='auto')
    im1 = axes[0,0].imshow(sig_act, **args)
    fig.colorbar(im1, ax=axes[0,0])
    im2 = axes[0,1].imshow(spl_act, **args)
    fig.colorbar(im2, ax=axes[0,1])

    args = dict(args, cmap='jet', vmin=-50, vmax=50)

    im3 = axes[1,0].imshow(100*bias1, **args)
    fig.colorbar(im3, ax=axes[1,0])

    im4 = axes[1,1].imshow(100*bias2, **args)
    fig.colorbar(im4, ax=axes[1,1])

    axes[0,0].set_title(f'signal {nsigma=}')
    axes[0,1].set_title(f'splat {nsigma=}')

    axes[1,0].set_title(f'splat/signal - 1 [%]')
    axes[1,1].set_title(f'signal/splat - 1 [%]')

    chan_tit = 'chans (rel)'
    tick_tit = 'ticks (rel)'
    axes[0,0].set_ylabel(chan_tit)
    axes[1,0].set_ylabel(chan_tit)
    axes[1,0].set_xlabel(tick_tit)
    axes[1,1].set_xlabel(tick_tit)

    fig.subplots_adjust(right=0.85)
    plt.tight_layout()


@dataclasses.dataclass
class Metrics:
    '''Metrics about a signal vs splat'''

    neor: int = 0
    ''' Number of channels with activity in either the signal or splat (or both)
    and over which the rest are calculated.  This can be less than the number of
    channels in the original "activity" arrays if any given channel has zero
    activity in both "signal" and "splat".  '''

    ineff: float = -1
    ''' The relative inefficiency.  This is the fraction of channels with splat
    but with zero signal.  '''

    fit: BaselineNoise | None = None
    '''
    Gaussian fit to relative difference.  .mu is bias and .sigma is resolution.
    '''


def calc_metrics(spl_qch, sig_qch, nbins=50):
    '''Return Metrics instance for splat and signal "channel activity" arrays.
    - spl_qch :: 1D array giving total charge per channel from splat
    - sig_qch :: 1D array giving total charge per channel from signala
    - nbins :: the number of bins over which to fit the relative difference.
    '''

    nspl = len(spl_qch)
    nsig = len(sig_qch)

    if nspl != nsig:
        raise ValueError(f'length mismatch {nspl=} != {nsig=}')

    # either-or, exclude channels where both are zero
    eor   = numpy.logical_or (spl_qch  > 0, sig_qch  > 0)
    # both are nonzero
    both  = numpy.logical_and(spl_qch  > 0, sig_qch  > 0)
    # splat but no signal (under efficient)
    nosig = numpy.logical_and(spl_qch  > 0, sig_qch == 0)
    wsig  = sig_qch  > 0
    # signal but not splat (over efficient)
    nospl = numpy.logical_and(spl_qch == 0, sig_qch  > 0)
    wspl  = spl_qch  > 0

    neor = numpy.sum(eor)
    nboth = numpy.sum(both)
    # inefficiency
    ineff = numpy.sum(nosig)/numpy.sum(wspl)

    reldiff = (spl_qch[both] - sig_qch[both])/spl_qch[both]
    vrange = 0.01*nbins/2
    bln = baseline_noise(reldiff, nbins, vrange)

    return Metrics(neor, ineff, bln)

def plot_metrics(splat_signal_activity_pairs, nbins=50, title="", letters="UVW"):
    plt.clf()
    fig, axes = plt.subplots(nrows=2, ncols=3, sharey="row")
    for pln, (spl_qch, sig_qch) in enumerate(splat_signal_activity_pairs):

        try:
            m = calc_metrics(spl_qch, sig_qch, nbins)
        except:
            log.error(f'error: failed to get metric for {pln=} {spl_qch.size=} {sig_qch.size=} {nbins=} {title=}')
            log.debug(f'skipped splat:  {spl_qch=}')
            log.debug(f'skipped signal: {sig_qch=}')
            continue
        counts, edges = m.fit.hist
        model = gauss_func(edges[:-1], m.fit.A, m.fit.mu, m.fit.sigma)

        letter = letters[pln]

        ax1,ax2 = axes[:,pln]

        ax1.plot(sig_qch, label='signal')
        ax1.plot(spl_qch, label='splat')
        ax1.set_xlabel('chans (rel)')
        ax1.set_ylabel('electrons')
        ax1.set_title(f'{letter} ineff={100*m.ineff:.1f}%')
        ax1.legend()

        ax2.step(edges[:-1], counts, label='data')
        ax2.plot(edges[:-1], model, label='fit')
        ax2.set_title(f'mu={100*m.fit.mu:.2f}%\nsig={100*m.fit.sigma:.2f}%')
        ax2.set_xlabel('difference [%]')
        ax2.set_ylabel('counts')
        ax2.legend()

    if title:
        plt.suptitle(title)
    else:
        plt.suptitle('(splat - signal) / splat')
    plt.tight_layout()
