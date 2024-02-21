#!/usr/bin/env python3
'''
Main CLI to wirecell.plot.
'''

import click
from wirecell.util import ario, plottools
from wirecell.util.cli import log, context, jsonnet_loader, frame_input, image_output
from wirecell.util import jsio
from wirecell.util.functions import unitify, unitify_parse
from wirecell import units
from pathlib import Path
import numpy
import matplotlib.pyplot as plt

@context("plot")
def cli(ctx):
    '''
    wirecell-plot command line interface
    '''
    pass


@cli.command("ntier-frames")
@click.option("-o", "--output", default="ntier-frames.pdf",
              help="Output file")
@click.option("-c", "--cmap",
              multiple=True,
              help="Give color map as tier=cmap")
@click.argument("files", nargs=-1)
def ntier_frames(cmap, output, files):
    '''
    Plot a number of per tier frames.

    Each file should a "frame file" (an ario stream of
    frame/channels/tickinfo arrays)
    '''
    if not files:
        raise click.BadParameter('no input files given')

    if output.endswith("pdf"):
        log.info(f'Saving to pdf: {output}')
        Outer = plottools.PdfPages
    else:
        log.info(f'Saving to: {output}')
        Outer = plottools.NameSequence

    cmaps = {kv[0]:kv[1] for kv in [x.split("=") for x in cmap]}

    readers = [ario.load(f) for f in files]

    tiers = list()
    idents = set()
    for reader in readers:
        myids = set()
        for key in reader:
            name, ident = key.split('.',1)[0].split('_')[1:]
            myids.add(ident)
        tiers.append(name)
        if not idents:
            idents = myids
        else:
            idents = idents.intersection(myids)
    idents = list(idents)
    idents.sort()

    ntiers = len(tiers)
    with Outer(output) as out:
        for ident in idents:
            for tier, reader in zip(tiers, readers):
                fig, ax = plt.subplots(nrows=1, ncols=1) # , sharex=True)

                vmin=-25
                vmax=+25
                if tier.startswith(("gauss", "wiener")):
                    vmin=-0
                    vmax=+10000

                aname = f'frame_{tier}_{ident}'
                try:
                    arr = reader[aname]
                except KeyError:
                    log.warn(f'No such key "{aname}".  Have: {len(reader)}')
                    log.warn(' '.join(reader.keys()))
                    continue
                log.debug(aname, arr.shape)
                arr = (arr.T - numpy.median(arr, axis=1)).T
                cmap = cmaps.get(tier, "viridis")
                im = ax.imshow(arr, aspect='equal', interpolation='none', cmap=cmap)
                plt.title(tier)
                plt.xlabel("time samples")
                plt.ylabel("channel IDs")
                plt.colorbar(im, ax=ax)
                out.savefig(fig)

    

@cli.command("frame")
@click.option("-n", "--name", default="wave",
              type=click.Choice(["wave","spectra"]),
              help="The frame plot type name [default=wave]")
@click.option("-t", "--tag", default="orig",
              help="The frame tag")
@click.option("-u", "--unit", default="ADC",
              help="The color units")
@click.option("--interactive", is_flag=True, default=False,
              help="running in interactive mode")
@image_output
@click.argument("datafile")
def frame(name, unit, tag, interactive, datafile, output, **kwds):
    '''
    Visualize a WCT frame with a plotter of a given name.
    '''
    from . import frames
    mod = getattr(frames, name)

    dat = ario.load(datafile)

    with output as out:
        mod(dat, out, tag, unit, interactive=interactive, **kwds)


@cli.command("comp1d")
@click.option("-n", "--name", default="wave",
              type=click.Choice(["wave", "spec"]),
              help="wave or spec")
@click.option("-t", "--tier", default="orig",
              help="orig, gauss, ...")
@click.option("-f", "--frames", default=None,
              help="instead of a tier selector, give comma-separated list of frame array names")
@click.option("--chmin", type=int, default=0,
              help="min channel, included")
@click.option("--chmax", type=int, default=0,
              help="max channel, not included")
@click.option("-u", "--unit", default="ADC",
              help="The color units")
@click.option("-x", "--xrange", type=(float, float), default=None,
              help="tick range of the output")
@click.option("-s", "--single", is_flag=True, default=False,
              help="force a single plot without file name mangling")
@click.option("--transform", multiple=True,
              type=click.Choice(["median","mean","ac","none"]), 
              help="type of data transformations")
@click.option("--interactive", is_flag=True, default=False,
              help="running in interactive mode")
@click.option("-o", "--output", type=click.Path(exists=False, dir_okay=False), required=True,
              help="The output file name, subject to mangling if not multipage format")
@click.option("--markers", default='o . , + X *', help="a space-separated list of markers")
@click.argument("datafiles", nargs=-1)
# @click.argument("output")
@click.pass_context
def comp1d(ctx, name, tier, frames, chmin, chmax, unit, xrange,
           single, transform, interactive, output, markers, datafiles):
    '''
    Compare waveforms from files

    FIXME: migrate to use of frame_input and image_output
    '''
    from .frames import comp1d as plotter
    if frames is None:
        frames = tier
    else:
        frames = [f.strip() for f in frames.split(",")]

    # too high and the pixel marker disappears.
    opts=dict(dpi=150)
    if single:
        out = plottools.NameSingleton(output, **opts)
    else:
        out = plottools.pages(output, **opts)

    markers=[m.strip() for m in markers.split(' ')]
    plotter(datafiles, out,
            name=name, frames=frames, chbeg=chmin, chend=chmax,
            unit=unit, xrange=xrange,
            interactive=interactive, transforms=transform,
            markers=markers)

@cli.command("channel-correlation")
@click.option("-t", "--tier", default="orig",
              help="orig, gauss, ...")
@click.option("--chmin", type=int, default=0,
              help="min channel, included")
@click.option("--chmax", type=int, default=0,
              help="max channel, not included")
@click.option("-u", "--unit", default="ADC",
              help="The color units")
@click.option("--interactive", is_flag=True, default=False,
              help="running in interactive mode")
@click.argument("datafile")
@click.argument("output")
@click.pass_context
def channel_correlation(ctx, tier, chmin, chmax, unit, interactive, datafile, output):
    '''
    Compare waveforms from files

    FIXME: migrate to use of frame_input and image_output
    '''
    from . import frames
    with plottools.pages(output) as out:
        frames.channel_correlation(datafile, out,
        tier=tier, chmin=chmin, chmax=chmax, unit=unit, interactive=interactive)



@cli.command("frame-diff")
@click.option("--style", type=click.Choice(["image", "axes"]), default="image")
@frame_input("1")
@frame_input("2")
@image_output
def frame_diff(array1, tickinfo1, channels1,
               array2, tickinfo2, channels2,
               style, output, **kwds):
    '''
    Take diff between two frames and write result as image
    '''
    if tickinfo1[1] != tickinfo1[1]:
        click.BadParameter(f'ticks must match, got {tickinfo1[1]} != {tickinfo1[1]}')
    tick = float(tickinfo1[1])
    if tick <= 0.0001:
        raise click.BadParameter(f'tick must be nonzero')

    import matplotlib.image
    from . import frames

    # fixme: make configurable
    channels = frames.common_channels(channels1, channels2)
    array1 = frames.select_channels(array1, channels1, channels)
    array2 = frames.select_channels(array2, channels2, channels)

    t1 = tickinfo1[0] + tickinfo1[2]*tick
    t2 = tickinfo2[0] + tickinfo2[2]*tick
    tmin = min(t1,t2)

    pad1 = int((t1-tmin)/tick)
    pad2 = int((t2-tmin)/tick)

    print(f'frame 1: pad={pad1} tickinfo={tickinfo1}')
    print(f'frame 2: pad={pad2} tickinfo={tickinfo2}')

    if pad1 > 0:
        array1 = numpy.pad(array1, ((0,0),(pad1,0)))
    elif pad1 < 0:
        array1 = array1[:,pad1:]

    if pad2 > 0:
        array2 = numpy.pad(array2, ((0,0),(pad2,0)))
    elif pad2 < 0:
        array2 = array2[:,pad2:]

    pad = array1.shape[1] - array2.shape[1]
    if pad > 0:
        array2 = numpy.pad(array2, ((0,0),(0,pad)))
    elif pad < 0:
        array1 = numpy.pad(array1, ((0,0),(0,-pad)))

    if array1.shape != array2.shape:
        raise click.UsageError("the programmer sucks")

    adiff = array1 - array2
    with output as out:
        plottools.image(adiff, style, interpolation='none', **plottools.imopts(**kwds))
        out.savefig()


@cli.command("frame-image")
@click.option("--transform", default='none',
              type=click.Choice(["median","mean","ac","none"]), 
              help="type of image transformations")
@click.option("--style", type=click.Choice(["image", "axes"]), default="image")
@frame_input()
@image_output
def frame_image(transform, style, array, output, aname, **kwds):
    '''
    Dump frame array to image
    '''
    import matplotlib.image
    from . import rebaseline
    tr = getattr(rebaseline, transform)
    array = tr(array)
    with output as out:
        plottools.image(array, style, **plottools.imopts(**kwds))
        out.savefig()


@cli.command("frame-means")
@frame_input()
@image_output
@click.option("--channels", default="800,800,960", help="Channels per plane")
def frame_means(array, channels, cmap, output, aname, ariofile, **kwds):
    '''
    Plot frames and their channel-wise and tick-wise means
    '''
    from wirecell.util.channels import parse_range
    channels = parse_range(channels)
    from . import frames
    with output as out:
        frames.frame_means(array, channels, cmap, aname, ariofile.path)
        out.savefig()


@cli.command("digitizer")
@image_output
@jsonnet_loader("jsiofile")
def digitzer(output, jsiofile, **kwds):
    '''
    Plots with output JSON file from test_digitizer

    See gen/test/test_digitizer_pdsp.cxx
    '''
    fadc = numpy.array(jsiofile["adc"])
    adc = {
        "float": fadc,
        "round": numpy.round(fadc),
        "floor": numpy.floor(fadc)}
    volts = numpy.array(jsiofile["volts"])
    
    with output as out:

        fig, axes = plt.subplots(2,1, figsize=(10,6))

        num = 25

        plt.suptitle("Digitizer with round vs floor")
        markers = ['o','P','X']
        def plot_slice(ax, slc):
            for ind, (key,val) in enumerate(adc.items()):
                ax.plot(volts[slc], val[slc], markers[ind]+'-', alpha=0.3, label=key)
            ax.legend()
            ax.set_xlabel("voltage [V]")
            ax.set_ylabel("ADC")
        plot_slice(axes[0], slice(0,25))
        plot_slice(axes[1], slice(-25,volts.size))

        plt.tight_layout()
        out.savefig()


@cli.command("channels")
@click.option("-c", "--channel", multiple=True, default=(), required=True,
              help="Specify channels, eg '1,2:4,5' are 1-5 but not 4")
@click.option("-t", "--trange", default=None, type=str,
              help="limit time range, eg '0,3*us'")
@click.option("-y", "--yrange", default=None, type=str,
              help="limit y-axis range in raw numbers, default is auto range")
@click.option("-f", "--frange", default=None, type=str,
              help="limit frequency range, eg '0,100*kHz'")
@image_output
@click.argument("frame_files", nargs=-1)
def channels(output, channel, trange, frange, yrange, frame_files, **kwds):
    '''
    Plot channels from multiple frame files.

    Frames need not have same sample period (tick).

    If --single put all on one plot, else per-channel plots are made
    '''

    from wirecell.util.frames import load as load_frames

    if trange:
        trange = unitify_parse(trange)
    if frange:
        frange = unitify_parse(frange)
    if yrange:
        yrange = unitify_parse(yrange)

    channels = list()
    for chan in channel:
        for one in chan.split(","):
            if ":" in one:
                f,l = one.split(":")
                channels += range(int(f), int(l))
            else:
                channels.append(int(one))

    # fixme: move this mess out of here

    frames = {ff: list(load_frames(ff)) for ff in frame_files}

    with output as out:

        for chan in channels:

            fig,axes = plt.subplots(nrows=1, ncols=2)
            fig.suptitle(f'channel {chan}')

            for fname, frs in frames.items():
                stem = Path(fname).stem
                for fr in frs:
                    wave = fr.waves(chan)
                    axes[0].set_title("waveforms")
                    axes[0].plot(fr.times/units.us, wave, drawstyle='steps')
                    if trange:
                        axes[0].set_xlim(trange[0]/units.us, trange[1]/units.us)
                    if yrange:
                        print(yrange)
                        axes[0].set_ylim(*yrange)
                    else:
                        plottools.rescaley(axes[0], fr.times/units.us, wave,
                                           (trange[0]/units.us, trange[1]/units.us))
                    axes[0].set_xlabel("time [us]")
                        

                    axes[1].set_title("spectra")
                    axes[1].plot(fr.freqs_MHz, numpy.abs(numpy.fft.fft(wave)),
                                 label=f'{fr.nticks}x{fr.period/units.ns:.0f}ns\n{stem}')
                    axes[1].set_yscale('log')
                    if frange:
                        axes[1].set_xlim(frange[0]/units.MHz, frange[1]/units.MHz)
                    else:
                        axes[1].set_xlim(0, fr.freqs_MHz[fr.nticks//2])
                    axes[1].set_xlabel("frequency [MHz]")
                    axes[1].legend()
                    print(fr.nticks, fr.period/units.ns, fr.duration/units.us)


            if not out.single:
                out.savefig()
                plt.clf()
        if out.single:
            out.savefig()



def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
