#!/usr/bin/env python
'''Make plots related to the "morse" pattern of depos.

See comments in wirecell.gen.morse for details.

'''

import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from wirecell.util import units
from wirecell.gen.morse import scale_slice, patch_chan_mask, gauss

def width_plots(out, signal, fps, channel_ranges, tick):
    '''
    A family of plots and fitting to measure peak widths.

    - out :: a PdfPages like object to plot to
    - signal :: a Nch x Ntick frame spanning channels in wire-attachment-order.
    - fps :: as returned by wirecell.gen.morse.frame_peaks()
    - channel_ranges :: a list of channels demarking planes
    - tick :: the sampling period
    '''
    # fixme: refactor this into multiple functions....

    chan_iota = numpy.arange(channel_ranges[0], channel_ranges[-1])
    tick_iota = numpy.arange(signal.shape[1])

    fig,ax = plt.subplots(1,1)
    plt.title('signal from "morse" depo pattern')
    max_tick = signal.shape[1]//2
    full_tick_mask = slice(0, max_tick)
    full_extent = (0, max_tick*tick/units.us,
                   chan_iota[0], chan_iota[-1])
    im = ax.imshow(signal[:,full_tick_mask], extent=full_extent, origin="lower",
                   aspect='auto',cmap='hot_r', vmax=1000)
    plt.colorbar(im, ax=ax)
    for letter, fp in zip("UVW",fps):
        cmask = scale_slice(patch_chan_mask(fp), 0.5)
        tmask = scale_slice(fp.total.mask, 2.0)
        rect = patches.Rectangle((tmask.start*tick/units.us, cmask.start),
                                 (tmask.stop-tmask.start)*tick/units.us,
                                 cmask.stop-cmask.start, color='blue', fill=False)
        ax.add_patch(rect)
        ax.text(tmask.stop*tick/units.us, cmask.stop, f'{letter} patch',
                horizontalalignment='center',
                verticalalignment='bottom')
    ax.set_xlabel("time [us]")
    ax.set_ylabel("channel")
    out.savefig()
    plt.clf()

    for plane, (ch1,ch2) in enumerate(zip(channel_ranges[:-1],channel_ranges[1:])):
        letter = "UVW"[plane]

        fp = fps[plane]

        full_chan_mask = scale_slice(patch_chan_mask(fp), 0.1)
        tick_mask = fp.total.mask

        extent = (tick_mask.start*tick/units.us, tick_mask.stop*tick/units.us,
                  full_chan_mask.start, full_chan_mask.stop)
        patch = signal[full_chan_mask, tick_mask]

        # The activity in channel vs time
        plt.title(f'{letter} patch')
        plt.imshow(patch, extent=extent, origin="lower", aspect='auto',cmap='hot_r')
        plt.xlabel("time [us]")
        plt.ylabel("channel")
        plt.colorbar()
        out.savefig()
        plt.clf()

        # Plot over channel direction
        plane_signal = numpy.zeros_like(signal)
        plane_signal[full_chan_mask, tick_mask] = patch

        chan_wave = plane_signal[:,fp.total.peak]
        plt.step(chan_iota[full_chan_mask], chan_wave[full_chan_mask])
        totwidth = 0
        for c in fp.chan:
            if c.mu is None:
                continue
            g = gauss(chan_iota, c.A, c.mu, c.sigma)
            lab=f'{c.sigma:.3f}'
            totwidth += c.sigma
            plt.plot(chan_iota[full_chan_mask], g[full_chan_mask], label=lab)
        plt.legend()
        totwidth = totwidth/len(fp.chan)
        plt.title(f'{letter} per-impact transverse widths, mean={totwidth:.3f}*pitch')
        plt.xlabel('channel')
        out.savefig()
        plt.clf()

        # Plot over time direction
        tick_mask = fp.total.mask
        tot_wave = numpy.sum(plane_signal, axis=0)
        time = tick_iota[tick_mask]*tick/units.us
        plt.step(time, tot_wave[tick_mask])
        g = gauss(tick_iota, fp.total.A, fp.total.mu, fp.total.sigma)
        sigma_us = fp.total.sigma*tick/units.us
        lab=f'{sigma_us:.3f} us'
        plt.plot(time, g[tick_mask], label=lab)
        plt.xlabel("time [us]")
        plt.title(f'{letter} total longitudinal width')
        plt.legend()
        out.savefig()
        plt.clf()

        fig,axes = plt.subplots(nrows=len(fp.tick), ncols=1, sharex=True,sharey=True)
        for ax,twp,cwp in zip(axes, fp.tick, fp.chan):
            ch = cwp.peak
            wave = plane_signal[ch]
            ax.step(time, wave[tick_mask])
            if twp.mu is None:
                continue    # failed fit
            g = gauss(tick_iota, twp.A, twp.mu, twp.sigma)
            lab=f'sig={twp.sigma*tick/units.us:.3f} us'
            ax.plot(time, g[tick_mask], label=lab)
            ax.legend()
        ax.set_xlabel('time [us]')
        plt.suptitle(f'{letter} per impact longitudinal widths, mean={sigma_us:.3f} us')
        fig.tight_layout()
        out.savefig()
        plt.clf()

