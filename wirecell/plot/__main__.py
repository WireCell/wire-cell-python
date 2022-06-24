#!/usr/bin/env python3
'''
Main CLI to wirecell.plot.
'''

import click
from wirecell.util import ario, plottools
import numpy
import matplotlib.pyplot as plt

cmddef = dict(context_settings = dict(help_option_names=['-h', '--help']))

@click.group(**cmddef)
@click.pass_context
def cli(ctx):
    '''
    wirecell-plot command line interface
    '''
    ctx.ensure_object(dict)


def good_cmap(diverging=True, color=True):
    '''
    An opinionated selection of the available colormaps.

    Diverging picks a cmap with central value as white, o.w. zero is
    white.  If color is False then a grayscale is used.

    https://matplotlib.org/stable/tutorials/colors/colormaps.html
    '''
    if diverging:
        if color: return "seismic"
        return "seismic"        # no gray diverging?
    if color: return "Reds"
    return "Greys"
    
def tier_cmap(tier, color=True):
    '''
    Return a good color map for the given data tier
    '''
    for diverging in ('orig', 'raw'):
        if tier.startswith(diverging):
            return good_cmap(True, color)
    for sequential in ('gauss', 'wiener'):
        if tier.startswith(sequential):
            return good_cmap(False, color)
    return good_cmap()


@cli.command("ntier-frames")
@click.option("--cmap", default="seismic",
              help="Set the color map")
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
                arr = (arr.T - numpy.median(arr, axis=1).T).T
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
@click.option("-t", "--tag", default="*",
              help="The frame tag")
@click.option("-u", "--unit", default="ADC",
              help="The color units")
@click.option("-r", "--range", default=25.0, type=float,
              help="The color range")
@click.argument("datafile")
@click.argument("output")
@click.pass_context
def frame(ctx, name, tag, unit, range, datafile, output):
    '''
    Make frame plots of given type.
    '''
    import wirecell.plot.frames
    mod = getattr(wirecell.plot.frames, name)
    dat = ario.load(datafile)
    with plottools.pages(output) as out:
        mod(dat, out, tag, unit, range)

    


def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
