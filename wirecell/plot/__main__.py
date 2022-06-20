#!/usr/bin/env python3
'''
Main CLI to wirecell.plot.
'''

import click
from wirecell.util import ario, plottools
import numpy
import matplotlib.pyplot as plt

@click.group()
@click.pass_context
def cli(ctx):
    '''
    wirecell-plot command line interface
    '''
    ctx.ensure_object(dict)



@cli.command("ntier-frames")
@click.option("--cmap", default="seismic",
              help="Set the color map")
@click.option("-o", "--output", default="ntier-frames.pdf",
              help="Output file")
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
                im = ax.imshow(arr, aspect='equal', interpolation='none',
                               cmap=cmap, vmin=vmin, vmax=vmax)
                plt.colorbar(im, ax=ax)
                out.savefig(fig)

    

@cli.command("frame")
@click.option("-n", "--name", default="wave",
              help="The frame plot name")
@click.argument("datafile")
@click.argument("output")
@click.pass_context
def frame(ctx, name, datafile, output):
    '''
    Plot per channel spectra for frame file
    '''
    import wirecell.plot.frames
    mod = getattr(wirecell.plot.frames, name)
    dat = ario.load(datafile)
    with plottools.pages(output) as out:
        mod(dat, out)

    


def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
