#!/usr/bin/env python
'''The simple splat / sim+SP (ssss) test is used, in part, to reproduce the
signal biase, efficiency and resolution metric used in the MicroBooNE SP-1
paper.

'''

import dataclasses
import numpy

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

    # Assure channels are ordered.  This is a sanity check but not truly a
    # generic one.  Some future detector may come along and decide to number
    # channel IDs in some way other than the monotonically increasing order of
    # contemporary detectors.  Such future detector will likely break
    # assumptions buried in a lot of code so might as well make that explicit
    # here.
    c2 = numpy.array(c)
    numpy.sort(c2)
    if not numpy.all(c == c2):
        raise ValueError("frame does not have ordered channel IDs which violates implicit convention")

    cmin = numpy.min(c)
    cmax = numpy.max(c)
    nch = cmax-cmin+1
    ff = numpy.zeros((nch, f.shape[1]), dtype=f.dtype)
    for irow, ch in enumerate(c):
        ff[ch-cmin] = f[irow]
    origin = "lower"            # row 0 = cmin at bottom of plot
    array_t0 = t0 + tbin*tick
    array_tf = array_t0 + ff.shape[1]*tick

    if trange:
        dt0 = (trange[0] - array_t0)/tick
        dtf = (trange[1] - array_tf)/tick
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

    array_extent = (array_t0, array_tf, cmin, cmax+1)

    return Frame(fname, ff, array_extent, origin, tick)


def align_channel_ranges(fr1, fr2):
    '''Return copies of fr1 and fr2 padded to cover the union of their channel
    ranges.  After alignment both frames have identical channel extent and the
    same number of rows, so that row i in fr1 corresponds to the same channel
    as row i in fr2.

    With origin="lower", row 0 holds the lowest channel (cmin) and row nch-1
    holds the highest channel (cmax).  Extending below the current cmin
    prepends zero rows; extending above the current cmax appends zero rows.
    '''
    _, _, c0_1, cf_1 = fr1.extent
    _, _, c0_2, cf_2 = fr2.extent

    c0 = min(c0_1, c0_2)   # new common cmin
    cf = max(cf_1, cf_2)   # new common cmax+1

    def _pad(fr, c0, cf):
        t0, tf, c0_fr, cf_fr = fr.extent
        pre  = c0_fr - c0    # rows to prepend (channels below c0_fr)
        post = cf - cf_fr    # rows to append  (channels above cf_fr-1)
        arr = fr.frame
        if pre > 0:
            arr = numpy.vstack([numpy.zeros((pre,  arr.shape[1]), dtype=arr.dtype), arr])
        if post > 0:
            arr = numpy.vstack([arr, numpy.zeros((post, arr.shape[1]), dtype=arr.dtype)])
        return Frame(fr.filename, arr, (t0, tf, c0, cf), fr.origin, fr.tick)

    if c0 == c0_1 and cf == cf_1 and c0 == c0_2 and cf == cf_2:
        return fr1, fr2
    return _pad(fr1, c0, cf), _pad(fr2, c0, cf)


def active_time_window(arrays, extent, tick, qlo=5e-3, qhi=0.995, pad_frac=0.1):
    '''Return (t0,tf) in system-of-units bounding the ticks holding the bulk of
    the activity of the given frame arrays.

    - arrays :: iterable of 2D (channel, tick) arrays sharing the given extent.
    - extent :: (t0, tf, cmin, cmax+1) as on a Frame.
    - tick :: sample period in system of units.
    - qlo, qhi :: the window brackets this quantile range of the cumulative
      |value| profile.  A cumulative (energy) window is used rather than a simple
      per-tick threshold so scattered low-level ringing far from the signal does
      not widen the window back out to the whole readout.
    - pad_frac :: pad the found window by this fraction of its width on each side.

    Falls back to the full extent if no activity is found.
    '''
    t0, tf, _, _ = extent
    prof = None
    for arr in arrays:
        col = numpy.abs(arr).sum(axis=0)
        prof = col if prof is None else prof + col
    if prof is None or prof.size == 0:
        return (t0, tf)
    total = prof.sum()
    if total <= 0:
        return (t0, tf)
    cum = numpy.cumsum(prof)
    lo = int(numpy.searchsorted(cum, qlo*total))
    hi = int(numpy.searchsorted(cum, qhi*total))
    if hi <= lo:
        hi = min(prof.size - 1, lo + 1)
    pad = int(round(pad_frac*(hi - lo + 1)))
    lo = max(0, lo - pad)
    hi = min(prof.size - 1, hi + pad)
    return (t0 + lo*tick, t0 + (hi + 1)*tick)


def plot_frame(gs, fr, channel_ranges=None, which="splat", tit="", channel_offset=0,
               xlim_us=None):
    '''
    Plot one Frame as 2D and 2x1D projections.

    If xlim_us is given as a (t0,tf) pair in microseconds, the time axis (and thus
    the 2D and time-projection views) are restricted to that window.
    '''
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    tunits = units.us

    # c0/cf are absolute channel IDs
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

    # Diverging "seismic" map centred at zero: white=0, red=positive, blue=the
    # rare negative signal.  Symmetric limits keep zero pinned to white.
    vext = 500
    im = fax.imshow(fr.frame, extent=extent_us, origin=fr.origin,
                    aspect='auto', vmin=-vext, vmax=vext, cmap='seismic',
                    interpolation='none')

    # Anchor per-plane labels within the visible time window so they stay on the
    # 2D axes even when the time axis is cropped to the activity.
    tlo_us, thi_us = xlim_us if xlim_us is not None else (t0_us, tf_us)

    tval = fr.frame.sum(axis=0)
    t = numpy.linspace(t0_us, tf_us, fr.frame.shape[1]+1,endpoint=True)
    tax.plot(t[:-1], tval)      # all channels
    if channel_ranges:
        for p, chans in zip("UVW",channel_ranges): # fixme: map to plane labels is only an assumption!
            # print(f'{p}: {chans=}')
            val = fr.frame[chans,:].sum(axis=0)
            c1 = chans.start + channel_offset
            c2 = chans.stop + channel_offset
            tax.plot(t[:-1], val, label=p)
            fax.plot([t0_us,tf_us], [c1,c1])
            fax.text(tlo_us + 0.05*(thi_us-tlo_us), c1 + 0.5*(c2-c1), p)
        fax.plot([t0_us,tf_us], [c2-1,c2-1])
        tax.legend()
    
    cval = fr.frame.sum(axis=1)
    c = numpy.linspace(fr.extent[2],fr.extent[3],fr.frame.shape[0]+1,endpoint=True)
    cax.plot(cval, c[:-1])

    if xlim_us is not None:
        fax.set_xlim(*xlim_us)      # tax shares x with fax

    return im

def plot_frames(spl, sig, channel_ranges, title="", channel_offset=0):
    '''
    Plot the two Frame objects spl (splat) and sig (signal).

    Channel ranges gives list of pair of channel min/max to interpret as
    contiguous rows on the Frame.array.

    Channel offset gives an offset from indices to IDENT numbers for labeling
    plots.
    '''
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    fig = plt.figure()
    pgs = GridSpec(1,2, figure=fig, width_ratios = [7,0.2])
    gs = GridSpecFromSubplotSpec(2, 1, pgs[0,0])

    # Restrict the time axis to where either frame has activity so the (usually
    # brief) signal is not lost in a mostly-empty readout window.  Both frames
    # share the same time extent, so use a common window.
    tw = active_time_window([spl.frame, sig.frame], spl.extent, spl.tick)
    xlim_us = (tw[0]/units.us, tw[1]/units.us)

    im1 = plot_frame(gs[0], spl, channel_ranges, which="splat",
                     channel_offset=channel_offset, xlim_us=xlim_us)
    im2 = plot_frame(gs[1], sig, channel_ranges, which="signal",
                     channel_offset=channel_offset, xlim_us=xlim_us)
    fig.colorbar(im2, cax=plt.subplot(pgs[0,1]))
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    
def plot_plane(spl_act, sig_act, nsigma=3.0, title=""):
    '''
    Plot splat and signal activity for one plane.

    '''
    import matplotlib.pyplot as plt

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


def plot_channels(spl, sig, ch, bbox, letter="", title="", channel_offset=0,
                  fracs=(0.25, 0.50, 0.75), tick_pad_frac=0.25):
    '''Plot signal and splat waveforms for a few channels sampled along a track.

    One page of len(fracs) rows.  Each row is the waveform (vs time) of both the
    splat and signal frames for a single channel taken at the given fractional
    position along the channel span of the track activity.

    - spl, sig :: the full (aligned) splat and signal Frame objects.
    - ch :: the plane's channel slice into the frame arrays.
    - bbox :: (channel_slice, tick_slice) of the biggest splat plateau, in
      plane-local coordinates (as returned by select_activity()/plateaus()).
    - letter :: plane label for titles.
    - channel_offset :: added to a row index to form the channel ID for labels.
    - fracs :: fractional positions along the activity channel span to sample.
    - tick_pad_frac :: pad the plotted time window by this fraction of the
      activity tick span on each side.
    '''
    import matplotlib.pyplot as plt

    chan_slice, tick_slice = bbox[0], bbox[1]
    cstart, cstop = chan_slice.start, chan_slice.stop
    span = cstop - cstart

    ncols = spl.frame.shape[1]
    t0_us = spl.extent[0] / units.us
    tick_us = spl.tick / units.us
    t = t0_us + numpy.arange(ncols) * tick_us

    # Time window bounding the track activity, padded.
    tpad = int(round(tick_pad_frac * (tick_slice.stop - tick_slice.start)))
    tlo = max(0, tick_slice.start - tpad)
    thi = min(ncols, tick_slice.stop + tpad)
    xlim_us = (t0_us + tlo*tick_us, t0_us + thi*tick_us)

    fig, axes = plt.subplots(nrows=len(fracs), ncols=1, sharex=True)
    if len(fracs) == 1:
        axes = [axes]

    for ax, frac in zip(axes, fracs):
        # plane-local row -> absolute frame row
        row_local = cstart + int(round(frac * (span - 1))) if span > 1 else cstart
        abs_row = ch.start + row_local
        chan_id = abs_row + channel_offset

        ax.plot(t, sig.frame[abs_row, :], label='signal')
        ax.plot(t, spl.frame[abs_row, :], label='splat')
        ax.set_xlim(*xlim_us)
        ax.set_ylabel('electrons')
        ax.set_title(f'{letter}-plane chan {chan_id} ({int(round(100*frac))}% along track)')
        ax.legend()

    axes[-1].set_xlabel('time [us]')
    if title:
        plt.suptitle(title)
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

    if nspl == 0 or nsig == 0:
        raise ValueError(f'empty input: {nspl=} {nsig=}')
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

    if nboth == 0:
        raise ValueError(f'no channels exist where both signal {nsig=} and splat {nspl=} are non-zero')

    # inefficiency
    ineff = numpy.sum(nosig)/numpy.sum(wspl)

    reldiff = (spl_qch[both] - sig_qch[both])/(spl_qch[both]+sig_qch[both])
    vrange = 0.01*nbins/2
    # print(f'{vrange=}')
    # print(f'{spl_qch[both]=}')
    # print(f'{sig_qch[both]=}')
    # print(f'{reldiff=}')
    bln = baseline_noise(reldiff, nbins, vrange)

    return Metrics(neor, ineff, bln)

def plot_metrics(splat_signal_activity_pairs, nbins=50, title="", letters="UVW"):
    import matplotlib.pyplot as plt

    plt.clf()
    fig, axes = plt.subplots(nrows=2, ncols=3, sharey="row")
    for pln, (spl_qch, sig_qch) in enumerate(splat_signal_activity_pairs):

        spl_qtot = numpy.sum(spl_qch)
        sig_qtot = numpy.sum(sig_qch)

        for name, arr in [("splat", spl_qch), ("signal", sig_qch)]:
            tot = numpy.sum(arr)
            n = len(arr)
            mean = tot/n
            print(f'Plane: {pln} {name}: [{n}] mean={mean}\n{arr}')

        if spl_qtot == 0 or sig_qtot == 0:
            log.warn(f'Warning: skipping {pln=}: splat qtot={spl_qtot}, signal qtot={sig_qtot}')
            continue

        try:
            m = calc_metrics(spl_qch, sig_qch, nbins)
        except Exception as err:
            log.error(f'Metric error: {err}')
            log.error(f'error: failed to get metric for {pln=} {spl_qch.size=} {sig_qch.size=} {nbins=} {title=}')
            log.error(f'skipped splat:  {spl_qch=}')
            log.error(f'skipped signal: {sig_qch=}')
            raise
            #continue
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
        # The row-2 histogram fits the symmetric relative difference actually
        # computed in calc_metrics(), not (splat-signal)/splat.
        plt.suptitle('(splat - signal) / (splat + signal)')
    plt.tight_layout()
