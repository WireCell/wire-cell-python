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
@click.option("-o", "--output", default="ntier-frames.pdf",
              help="Output file")
@click.argument("files", nargs=-1)
def ntier_frames(output, files):
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

                aname = f'frame_{tier}_{ident}'
                arr = reader[aname]
                print(aname, arr.shape)
                arr = (arr.T - numpy.median(arr, axis=1).T).T
                im = ax.imshow(arr, aspect='equal', interpolation='none')
                plt.colorbar(im, ax=ax)
                out.savefig(fig)

    
def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
