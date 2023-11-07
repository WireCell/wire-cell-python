#!/usr/bin/env python3

import os
import sys
import math
import json
import click
import numpy
from collections import defaultdict
from wirecell import units
from wirecell.util.functions import unitify, unitify_parse
from wirecell.util.cli import context, log

@context("ls4gan")
def cli(ctx):
    '''
    Wire Cell Toolkit Utility Commands
    '''
    pass


@cli.command("npz-to-wct")
@click.option("-T", "--transpose", default=False, is_flag=True,
              help="Transpose input arrays to give per-channel rows")
@click.option("-o", "--output", type=str,
              help="Output image file")
@click.option("-n", "--name", default="",
              help="The name tag for the output arrays")
@click.option("-f", "--format", default="frame",
              type=click.Choice(["frame",]), # "tensor"
              help="Set the output file format")
@click.option("-t", "--tinfo", type=str,
              default="0,0.5*us,0",
              help="The tick info list: time,tick,tbin")
@click.option("-b", "--baseline", default=0.0,
              help="An additive, prescaled offset")
@click.option("-s", "--scale", default=1.0,
              help="A multiplicative scaling")
@click.option("-d", "--dtype", default="i2",
              type=click.Choice(["i2","f4"]),
              help="The data type of output samples in Numpy dtype form")
@click.option("-c", "--channels", default=None,
              help="Channel specification")
@click.option("-e", "--event", default=0,
              help="Event count start")
@click.option("-z", "--compress", default=True, is_flag=True,
              help="Whether to compress if output file is .npz")
@click.argument("npzfile")
def npz_to_wct(transpose, output, name, format, tinfo, baseline, scale, dtype, channels, event, compress, npzfile):
    """Convert a npz file holding 2D frame array(s) to a file for input to WCT.

    A linear transform and type cast is be applied to the input
    samples prior to output:

        output = dtype((input + baseline) * scale)

    Channel ID numbers for rows of the input array must be specified
    in a way to that matches the target detector.  They may be
    specified in a number of ways:

    - Default (unspecified) will number them starting at ID=0.
    - A single integer N will number them starting at ID=N.
    - A comma-separated list: 1,2,3,.... exaustively gives all IDs.
    - A file.npy with a 1D array of integers.
    - A file.npz:array_name with a 1D array of integers.

    """
    from collections import OrderedDict

    tinfo = unitify(tinfo)
    baseline = float(baseline)
    scale = float(scale)

    out_arrays = OrderedDict()
    event = int(event)          # count "event" number
    fp = numpy.load(npzfile)
    for aname in fp:
        arr = fp[aname]
        print(f'processing {npzfile}')
        if transpose:
            arr = arr.T
        if len(arr.shape) != 3:
            raise click.BadParameter(f'input array {aname} wrong shape: {arr.shape}')
        arr = numpy.vstack((arr[0:800,:,0],arr[0:800,:,1],arr[0:960,:,2]))

        nchans = arr.shape[0]

        # figure out channels in the loop as nchans may differ array
        # to array.
        if channels is None:
            channels = list(range(nchans))
        elif channels.isdigit():
            ch0 = int(channels)
            channels = list(range(ch0, ch0+nchans))
        elif "," in channels:
            channels = unitify(channels)
            if len(channels) != nchans:
                raise click.BadParameter(f'input array has {nchans} channels but given {len(channels)} channels')

        elif channels.endswith(".npy"):
            channels = numpy.load(channels)
        elif ".npz:" in channels:
            fname,cname = channels.split(":",1)
            cfp = numpy.load(fname)
            channels = cfp[cname]
        else:
            raise click.BadParameter(f'unsupported form for channels: {channels}')

        channels = numpy.array(channels, 'i4')

        label = f'{name}_{event}'
        event += 1

        out_arrays[f'frame_{label}'] = numpy.array((arr + baseline) * scale, dtype=dtype)
        out_arrays[f'channels_{label}'] = channels
        out_arrays[f'tickinfo_{label}'] = tinfo

    out_dir = os.path.dirname(output)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if output.endswith(".npz"):
        if compress:
            numpy.savez_compressed(output, **out_arrays)
        else:
            numpy.savez(output, **out_arrays)
    else:
        raise click.BadParameter(f'unsupported output file type: {output}')

def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
