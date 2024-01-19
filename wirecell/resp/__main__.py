#!/usr/bin/env python3
'''
Main CLI to wirecell.resp.
'''

import click
from wirecell.util.fileio import load as source_loader
from wirecell import units
from wirecell.util.functions import unitify
import numpy

cmddef = dict(context_settings = dict(help_option_names=['-h', '--help']))

@click.group(**cmddef)
@click.pass_context
def cli(ctx):
    '''
    wirecell-resp command line interface
    '''
    ctx.ensure_object(dict)


@cli.command("gf2npz")
@click.option("-o", "--output",
              type=click.Path(dir_okay=False, writable=True),
              help="Name the output NPZ file")
@click.option("--origin", type=str,
              help="Set drift origin (give units, eg '10*cm').")
@click.option("--speed", type=str,
              help="Set drift speed at start of response (give untis, eg '1.114*mm/us').")
@click.argument("dataset")
def gf2npz(output, origin, speed, dataset):
    '''
    Convert a Garfield data set to a "WCT response NPZ" file.
    '''
    if not all([speed, origin]):
        raise ValueError("You MUST give --speed and --origin")

    from wirecell.resp.garfield import (
        dataset_asdict, dsdict2arrays)
    source = source_loader(dataset, pattern="*.dat")
    ds = dataset_asdict(source)

    origin = eval(origin, units.__dict__)
    speed = eval(speed, units.__dict__)
    arrs = dsdict2arrays(ds, speed, origin)
    numpy.savez(output, **arrs)

@cli.command("gf-info")
@click.argument("dataset")
def gf_info(dataset):
    '''
    Give info about a garfield dataset
    '''
    from wirecell.resp.garfield import (
        dataset_asdict, dsdict_dump)
    source = source_loader(dataset, pattern="*.dat")
    ds = dataset_asdict(source)
    dsdict_dump(ds)


@cli.command("resample")
@click.option("-t", "--tick", default=None, type=str,
              help="Resample the field response to have this sample period with units, eg '64*ns'")
@click.option("-P", "--period", default=None, type=str,
              help="Override the sampling period given in the frfile, eg '100*ns'")
@click.option("-o", "--output", default="/dev/stdout",
              help="File in which to write the result")
@click.option("-e", "--error", default=1e-6,
              help="Precision by which integer and rationality conditions are judged")
@click.argument("frfile")
def resample(tick, period, output, error, frfile):
    '''Resample the FR.

    The initial sampling period Ts (fr.period or --period if given) and the
    resampled period Tr (--tick) must satisfy the LMN rationality condition.

    The total duration of the resampled responses may change.

    See also:

    - wirecell-util lmn 
    - wirecell-sigproc fr2npz 

    '''

    import wirecell.sigproc.response.persist as per
    import wirecell.resp.resample as res

    tick = unitify(tick)
    period = unitify(period)

    fr = per.load(frfile)
    fr = res.resample(fr, tick, period)

    per.dump(output, fr)


def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
