#!/usr/bin/env python3
'''
Main CLI to wirecell.plot.
'''

import click
from wirecell.util import ario, plottools
import numpy
import matplotlib.pyplot as plt
from .cli import frame_to_image

cmddef = dict(context_settings = dict(help_option_names=['-h', '--help']))

@click.group(**cmddef)
@click.pass_context
def cli(ctx):
    '''
    wirecell-plot command line interface
    '''
    ctx.ensure_object(dict)


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
    if output.endswith("pdf"):
        print(f'Saving to pdf: {output}')
        Outer = plottools.PdfPages
    else:
        print(f'Saving to: {output}')
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
                    print(f'No such key "{aname}".  Have: {len(reader)}')
                    print(' '.join(reader.keys()))
                    continue
                print(aname, arr.shape)
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
              help="The frame plot name")
@click.option("-t", "--tag", default="orig",
              help="The frame tag")
@click.option("-u", "--unit", default="ADC",
              help="The color units")
@click.option("-r", "--range", default=25.0, type=float,
              help="The color range")
@click.option("--interactive", is_flag=True, default=False,
              help="running in interactive mode")
@click.argument("datafile")
@click.argument("output")
@click.pass_context
def frame(ctx, name, tag, unit, range, interactive, datafile, output):
    '''
    Make frame plots of given type.
    '''
    from . import frames
    mod = getattr(frames, name)
    dat = ario.load(datafile)
    with plottools.pages(output) as out:
        mod(dat, out, tag, unit, range, interactive=interactive)


@cli.command("comp1d")
@click.option("-n", "--name", default="wave",
              help="wave or spec")
@click.option("-t", "--tier", default="orig",
              help="orig, gauss, ...")
@click.option("--chmin", type=int, default=0,
              help="min channel, included")
@click.option("--chmax", type=int, default=0,
              help="max channel, not included")
@click.option("-u", "--unit", default="ADC",
              help="The color units")
@click.option("-x", "--xrange", type=(float, float), default=None,
              help="tick range of the output")
@click.option("--baseline", default="median",
              type=click.Choice(["median","mean","none"]), 
              help="type of rebaselining procedure")
@click.option("--interactive", is_flag=True, default=False,
              help="running in interactive mode")
@click.argument("datafile1")
@click.argument("datafile2")
@click.argument("output")
@click.pass_context
def comp1d(ctx, name, tier, chmin, chmax, unit, xrange, baseline, interactive, datafile1, datafile2, output):
    '''
    Compare waveforms from files
    '''
    from . import frames
    with plottools.pages(output) as out:
        frames.comp1d(datafile1, datafile2, out,
                      name=name, tier=tier, chmin=chmin, chmax=chmax,
                      unit=unit, xrange=xrange,
                      interactive=interactive, baseline=baseline)

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
    '''
    from . import frames
    with plottools.pages(output) as out:
        frames.channel_correlation(datafile, out,
        tier=tier, chmin=chmin, chmax=chmax, unit=unit, interactive=interactive)


@cli.command("frame-image")
@frame_to_image
def frame_image(array, channels, cmap, format, output, aname, fname):
    '''
    Dump frame array to image, ignoring channels.
    '''
    import matplotlib.image

    matplotlib.image.imsave(output, array, format=format, cmap=cmap)

@cli.command("frame-means")
@frame_to_image
def frame_means(array, channels, cmap, format, output, aname, fname):
    '''
    Plot frames and their channel-wise and tick-wise means
    '''
    from . import frames
    frames.frame_means(array, channels, cmap, format, output, aname, fname)


def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
