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
@click.option("-r", "--ranges", nargs=6, type=click.Tuple([int, int, int, int, int, int]), 
              default=[0, 800, 0, 800, 0, 960],
              help="ubeg uend vbeg vend wbeg wend, end is not included")
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
@click.option("-rd", "--rounding", default="floor",
              type=click.Choice(["floor","round"]),
              help="How to round if dtype is integer")
@click.option("-c", "--channels", default=None,
              help="Channel specification")
@click.option("-e", "--event", default=0,
              help="Event count start")
@click.option("-z", "--compress", default=True, is_flag=True,
              help="Whether to compress if output file is .npz")
@click.argument("npzfile")
def npz_to_wct(transpose, output, name, format, ranges, tinfo, baseline, scale, dtype, rounding, channels, event, compress, npzfile):
    """Convert a npz file holding 3D frame array(s) to a file for input to WCT.
    assumes channel, tick, plane(3)

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
        # assume input is (channel, tick, plane(3))
        arr = numpy.vstack((arr[ranges[0]:ranges[1],:,0],arr[ranges[2]:ranges[3],:,1],arr[ranges[4]:ranges[5],:,2]))

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
        if rounding == "floor":
            out_arrays[f'frame_{label}'] = numpy.array((arr + baseline) * scale, dtype=dtype)
        elif rounding == "round":
            out_arrays[f'frame_{label}'] = numpy.array(numpy.round((arr + baseline) * scale), dtype=dtype)
        else:
            raise click.BadParameter(f'unsupported rounding: {rounding}')
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


@cli.command("comp-metric")
@click.option("-o", "--output", type=str,
              help="Output image file")
@click.option("-m", "--metric", type=str,
              help="L1 or L2")
@click.option("-f", "--format", type=str, default="2D",
              help="2D (WCT) or 3D (LS4GAN)")
@click.option("-r", "--ranges", nargs=6, type=click.Tuple([int, int, int, int, int, int]), 
              default=[0, 800, 800, 1600, 1600, 2560],
              help="ubeg uend vbeg vend wbeg wend, end is not included")
@click.option("-b", "--baseline", type=bool, default=False,
              help="Do baseline subtraction or not")
@click.option("-t", "--tag", default="orig",
              help="The frame tag")
@click.option("-d", "--dtype", default="int16",
              help="int16,float")
@click.argument("npzfile1")
@click.argument("npzfile2")
def comp_metric(output, metric, format, ranges, baseline, tag, dtype, npzfile1, npzfile2):
    """
    input: channel, tick, plane(3)
    output: metrics for 3 planes (m_u, m_v, m_w)
    """
    # print(f'processing {npzfile1} {npzfile2}')
    
    def get_dense_array(npzfile, tag=tag, dtype=dtype):
        from wirecell.util import ario
        dat = ario.load(npzfile)
        frame_keys = [f for f in dat.keys() if f.startswith('frame_')]
        frames = sorted([f for f in frame_keys if f.startswith(f'frame_{tag}')])
        if not frames:
            found = ', '.join(frame_keys)
            msg = f'No frames of tier "{tag}": found: {found}'
            raise IOError(msg)

        # TODO: implement uscale
        uscale = 1.
        # print(f'dtype: {dtype}')

        for fname in frames:
            _,tag,num = fname.split("_")
            # print(f'frame "{tag}" #{num}')
            ticks = dat[f'tickinfo_{tag}_{num}']
            chans = dat[f'channels_{tag}_{num}']
            chmin = numpy.min(chans)
            chmax = numpy.max(chans)
            nchan = chmax-chmin+1;

            waves = dat[fname]      # (nch x ntick)
            if baseline:
                waves = numpy.array((waves.T - numpy.median(waves, axis=1)).T, dtype=dtype)
            else:
                waves = numpy.array(waves, dtype=dtype)
            if dtype == float:
                waves /= 1.0*uscale
            chwaves = numpy.zeros((nchan, waves.shape[1]), dtype=dtype)
            for ind,ch in enumerate(chans):
                chwaves[ch-chmin] = waves[ind]
        return chwaves

    def L1(a, b):
        if a.shape != b.shape:
            raise ValueError("Input arrays have different shapes")
        if a.size == 0:
            raise ValueError("input arrays have zero elements")
        # print(a.shape, a.size)
        return numpy.sum(numpy.abs(a-b)) / a.size
    def L2(a, b):
        if a.shape != b.shape:
            raise ValueError("Input arrays have different shapes")
        if a.size == 0:
            raise ValueError("input arrays have zero elements")
        return numpy.linalg.norm(a-b) / numpy.sqrt(a.size)
    metric_func = {"L1": L1, "L2": L2}[metric]

    if format == "2D":
        arr1 = get_dense_array(npzfile1)
        arr2 = get_dense_array(npzfile2)
    elif format == "3D":
        fp1 = numpy.load(npzfile1)
        fp2 = numpy.load(npzfile2)
        for aname in fp1:
            arr1 = fp1[aname]
            arr2 = fp2[aname]
    else:
        raise ValueError(f'format {format} not supported')

    if arr1.shape != arr2.shape:
        raise ValueError("Input arrays have different shapes")

    output = []
    if format == "2D":
        # (channel, tick)
        output.append(metric_func(arr1[ranges[0]:ranges[1],:],arr2[ranges[0]:ranges[1],:]))
        output.append(metric_func(arr1[ranges[2]:ranges[3],:],arr2[ranges[2]:ranges[3],:]))
        output.append(metric_func(arr1[ranges[4]:ranges[5],:],arr2[ranges[4]:ranges[5],:]))
    elif format == "3D":
        # (channel, tick, plane(3))
        output.append(metric_func(arr1[ranges[0]:ranges[1],:,0],arr2[ranges[0]:ranges[1],:,0]))
        output.append(metric_func(arr1[ranges[2]:ranges[3],:,1],arr2[ranges[2]:ranges[3],:,1]))
        output.append(metric_func(arr1[ranges[4]:ranges[5],:,2],arr2[ranges[4]:ranges[5],:,2]))

    print(','.join(map(str, output)))
    return output

def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
