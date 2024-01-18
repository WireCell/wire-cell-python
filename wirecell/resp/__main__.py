#!/usr/bin/env python3
'''
Main CLI to wirecell.resp.
'''

import click
from wirecell.util.fileio import load as source_loader
from wirecell import units
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


@cli.command("field-response-transform")
@click.option("-g", "--gain", default=None, type=str,
                  help="Set gain with units, eg '14*mV/fC'")
@click.option("-s", "--shaping", default=None, type=str,
                  help="Set shaping time with units, eg '2*us'")
@click.option("-n", "--number", default=None, type=int,
              help="Resample the field response to have this number of ticks")
@click.option("-t", "--tick", default=None, type=str,
              help="Resample the field response to have this sample period with units, eg '64*ns'")
@click.option("-o", "--output", default="/dev/stdout",
              help="File in which to write the result")
@click.option("-s", "--schema", default="json.gz", 
              help="The format to assume given as an extension")
@click.option("-z", "--compression", default=True,
              help="Apply compression to the output")
@click.argument("frfile")
def field_response_transform(gain, shaping, number, tick, output, schema, frfile):
    '''Apply a transformation to a field response (FR) file.

    This may be used to transform FR file format or apply a resampling in time.

    If both gain and shaping are given then convolve with Cold Electronics
    response.

    If number or tick but not both are given then the duration of the input FR
    signals is held fixed in the resampling.  If both are given, the duration
    will made "number*tick".  Resampling must be "lmn-rational".

    See also: wirecell-sigproc fr2npz

    '''
    import wirecell.sigproc.response.persist as per
    import wirecell.sigproc.response.arrays as arrs

    fr = per.load(frfile)
    if gain and shaping:
        gain = unitify(gain)
        shaping = unitify(shaping)
        dat = arrs.fr2arrays(fr, gain, shaping)
    else:
        dat = arrs.fr2arrays(fr)

    if number or tick:
        tick = unitify(tick) if tick else None
        dat = arrs.resample(dat, tick, number)

    per.dump(output, dat, 

    numpy.savez(npz_file, **dat)

def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
