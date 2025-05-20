#!/usr/bin/env python3

import json
import click
import functools

from wirecell import units
from wirecell.util.functions import unitify, unitify_parse
from wirecell.util.cli import context, log, frame_input, image_output
from wirecell.util.plottools import pages

@context("gen")
def cli(ctx):
    '''
    Wire Cell Signal Simulation Commands
    '''
    pass

@cli.command("unitify-depos")
@click.option("-j", "--json_path", default='depos',
                  help="Data structure path to the deposition array in the input file.")
@click.option("-d", "--distance_unit", default='mm',
                  help="Set the unit of distance assumed in the input file (for x,y,z,s).")
@click.option("-t", "--time_unit", default='ns',
                  help="Set the unit of time assumed in the input file (for t).")
@click.option("-e", "--energy_unit", default='MeV',
                  help="Set the unit of energy assumed in the input file (for q).")
@click.option("-s", "--step_unit", default=None,
                  help="Set the unit of step, if different than distance (for s).")
@click.argument("input-file")
@click.argument("output-file")
@click.pass_context
def unitify_depos(ctx, json_path,
                      distance_unit, time_unit, energy_unit, step_unit,
                      input_file, output_file):
    '''
    Set units for a WCT JSON deposition file.

    The units given are what the input file should be assumed to
    follow.  The output file will then be in WCT's system of units.

    '''

    import depos as deposmod
    depos = deposmod.load(input_file)
    depos = deposmod.apply_units(depos, distance_unit, time_unit, energy_unit, step_unit);
    deposmod.dump(output_file, depos)


@cli.command("move-depos")
@click.option("-j", "--json_path", default='depos',
                  help="Data structure path to the deposition array in the input file.")
@click.option("-c", "--center", nargs=3,
                  help='Move deposition distribution to given x,y,z center. eg -c 1*m 2*cm 3*um')
@click.option("-o", "--offset", nargs=3,
                  help='Move deposition by vector offset. eg -c 1*m 2*cm 3*um')
@click.argument("input-file")
@click.argument("output-file")
@click.pass_context
def move_depos(ctx, json_path, center, offset,
                   input_file, output_file):
    '''
    Apply some transformations to a file of JSON depos and create a new file.
    '''
    import depos as deposmod
    depos = deposmod.load(input_file)

    if center:
        center = tuple([float(eval(c, units.__dict__)) for c in center])
        depos = deposmod.center(depos, center)

    if offset:
        offset = tuple([float(eval(c, units.__dict__)) for c in offset])
        depos = deposmod.move(depos, offset)
    deposmod.dump(output_file, depos)

@cli.command("plot-depos")
@click.option("-g", "--generation", default=0,
              help="The depo generation index. [default=0]")
@click.option("-i", "--index", default=0,
              help="The depos set index in the file. [default=0]")
@click.option("-p", "--plot", default=['qxz'], multiple=True,
              type=click.Choice("qxz qxy qzy qxt qzt t x xzqscat xyqscat tzqscat tyqscat".split(' ')),
              help="The plot(s) to make. [default=qxz, multiple okay]")
@click.option("-s", "--speed", default=None,
              help="Drift speed, with units eg '1.6*mm/us'. [default=None]")
@click.option("--t0", default="0*ns",
              help="Arbitrary additive time used in drift speed assignment, use units. [default=0*ns]")
@click.option("--title", default=None, type=str,
              help="Give plot title")
@click.argument("input-file")
#@click.argument("output-file")
@image_output
def plot_depos(generation, index, plot, speed, t0,
               title, input_file, output, **kwds):
    '''
    Make a plot from a WCT depo file.

    If speed is given, a depo.X is calculated as (time+t0)*speed and
    depo.T is untouched.

    Else, depo.T will have t0 added and depo.X untouched.

    Note, a t0 of the ductors "start_time" will generally bring depos
    into alignement with products for simulated frames.

    See also "wirecell-img paraview-depos".
    '''
    import wirecell.gen.depos as deposmod
    import matplotlib.pyplot as plt

    depos = deposmod.load(input_file, index, generation)
    if 't' not in depos or len(depos['t']) == 0:
        raise click.BadParameter(f'No depos for index={index} and generation={generation} in {input_file}')

    t0 = unitify(t0)
    if speed is not None:
        speed = unitify(speed)
        log.debug(f'applying speed: {speed/(units.mm/units.us)} mm/us')
        depos['x'] = speed*(depos['t']+t0)
    else:
        depos['t'] += t0

    with output as out:
        for plot_type in plot:
            plotter = getattr(deposmod, "plot_"+plot_type)
            plotter(depos, **kwds) # may throw
            if title:
                plt.suptitle(title)
            out.savefig()

@cli.command("plot-test-boundaries")
@click.option("-t", "--times", default=[100.0,105.0], type=float, nargs=2,
              help="Two range of times over which to limit frame plots, in ms.")
@click.argument("npz-file")
@click.argument("pdf-file")
@click.pass_context
def plot_test_boundaries(ctx, times, npz_file, pdf_file):
    '''
    Make some plots from the boundaries test.

        wire-cell -c gen/test/test_boundaries.jsonnet

    this makes a test_boundaries.npz file which is input to this command.
    '''
    from wirecell.gen import sim
    import matplotlib.pyplot as plt
    import numpy

    f = numpy.load(npz_file);

    fnums = [int(k.split('_')[-1]) for k in f.keys() if k.startswith("frame")]
    dnums = [int(k.split('_')[-1]) for k in f.keys() if k.startswith("depo_data")]

    with pages(pdf_file) as pdf:

        for fnum in fnums:
            fo = sim.Frame(f, fnum)

            fig, axes = fo.plot(t0=times[0]*units.ms, tf=times[1]*units.ms, raw=False)
            fig.suptitle("Frame %d" % fnum)
            pdf.savefig(fig)
            plt.close()

        for dnum in dnums:
            depo = sim.Depos(f, dnum)
            fig, axes = depo.plot()
            fig.suptitle("Depo group %d" % fnum)
            pdf.savefig(fig)
            plt.close()
        
@cli.command("plot-sim")
@click.argument("input-file")
@click.argument("output-file")
@click.option("--ticks/--no-ticks", default=False,
              help="Plot ticks, not time.")
@click.option("-p", "--plot", default='frame',
              help="The plot to make.")
@click.option("--tag", default='',
              help="The frame tag.")
@click.option("-t", "--time-range", default='',
              help="The time range in ms.")
@click.option("-n", "--number", default="0",
              help="One or more comma separated frame or depo indices.")
@click.option("-c", "--channel-groups", default='',
              help="Indices of channel groups as comma separated list.")
@click.option("-b", "--channel-boundaries", default='',
              help="Channels at which there are boundaries, eg 0,2560,5120 for 2 APAs")
@click.option("--dpi", default=600,
              help="Resolution of plots in dots per inch")
@click.pass_context
def plot_sim(ctx, input_file, output_file, ticks, plot, tag, time_range, number, channel_groups, channel_boundaries, dpi):
    '''
    Make plots of sim quantities saved into numpy array files.
    '''
    from wirecell.util import ario
    import wirecell.gen.sim
    from wirecell import units
    import matplotlib.pyplot as plt
    from wirecell.util.plottools import NameSequence

    if not time_range:
        if ticks:
            time_range = "0,-1"
        else:
            time_range = '0,5'

    
    fp = ario.load(input_file)

    numbers = [int(i.strip()) for i in number.split(",") if i.strip()]

    with pages(output_file) as out:
        
        # hack
        if len(numbers) == 1 and hasattr(out, "index"):
            out.index = None

        for onenum in numbers:
            if 'frame' in plot:
                fr = wirecell.gen.sim.Frame(fp, tag=tag, ident=onenum)

                if channel_groups:
                    ch = [ch[int(ci)] for ci in channel_groups.split(",")]
                elif channel_boundaries:
                    channel_boundaries = list(wirecell.gen.sim.parse_channel_boundaries(channel_boundaries))
                    ch = [[channel_boundaries.pop(0),],]
                    while channel_boundaries:
                        one = channel_boundaries.pop(0)
                        ch[-1].append(one-1)
                        if channel_boundaries:
                            ch.append([one])
                else:
                    ch = wirecell.gen.sim.group_channel_indices(fr.channels)
                log.debug ("Using channel groups: ", ch)

                if ticks:
                    plotter = fr.plot_ticks
                    t0,tf = [int(t,10) for t in time_range.split(",")]
                else:
                    plotter = fr.plot
                    t0,tf = [float(t)*units.ms for t in time_range.split(",")]

                fig, axes = plotter(t0, tf, raw=False, chinds=ch)
                out.savefig(fig, dpi=dpi)
                plt.close()

            if 'depo' in plot:

                deps = wirecell.gen.sim.Depos(fp, ident=onenum)
                fig, axes = deps.plot()
                out.savefig(fig, dpi=dpi)
                plt.close()

@cli.command("depo-morse")
@click.option("-L", "--length", default="1*mm",
              help="The length of each track")
@click.option("-S", "--step-size", default="0.1*mm",
              help="Distance between deposition of ionization electron groups")
@click.option("-T", "--time-jump", default="500*us",
              help="The time between groups of depos targeting each plane")
@click.option("-P", "--pitch-jump", default=25, type=int,
              help="The separation, in units of wire pitch, between each line of depos for a plane")
@click.option("-I", "--impact-jump", default=0.1, type=float,
              help="The separation of impact positions in units of wire pitch.")
@click.option("-R", "--response", default="10*cm", 
              help="The distance in global coordinates from the collection plane to the response plane")
@click.option("-e", "--electron-density", default="500/mm",
              help="Linear electron density on track (number of electrons per unit track length)")
@click.option("-w", "--wires", default="pdsp",
              help="The name of wires file or canonical detector name")
@click.option("--anode",  default=0, type=int, help="The anode to target")
@click.option("--face",   default=0, type=int, help="The face to target")
@click.option("--planes", default="0,1,2", help="The planes to target (default is all)")
@click.option("-o", "--output",
              type=click.Path(dir_okay=False, file_okay=True),
              help="Depo file which to save the results")
def depo_morse(length, step_size,
               time_jump, pitch_jump, impact_jump,
               response, electron_density,
               wires, anode, face, planes, output):
    '''Produce a set of depos in a pattern aligned to wires of each plane.

    All depos are placed at the response plane and have no Gaussian extent.

    There are three "planes" of depos with each "targeting" one wire plane.

    In a "plane" all depos have the same time and plane times are separated by
    the value given by the "-t/--time" option.

    Depos "target" a wire plane with a 2D rectangular grid of depo positions.
    One axis of the grid is parallel to the wires and the other is
    perpendicular, along the pitch direction.  Along the wires, depos are
    produced according to the valud of the -S/--step-size option.  Along the
    pitch, these lines of depos are advanced by the value provided by the value
    of the -P/--pitch option with an additional and progressive advance ment.
    The result a central line of depos being above the central wire of the plane
    and each subsequent neighbor advancing by one impact position as given by
    the value of the -I/--impact option.

    Each line shows up as a "dot" in the targeted plane view (thus the bad pun
    for the command name).  When such pattern of depos are run through WC
    simulation and signal processing the resulting signal widths measure a
    broadening incurred in addition to any due to difussion.  Their shape may be
    fit with a Gaussian model and the average sigma used with DepoFluxSplat to
    produce "splat" signals similar to signals from sim+SP.

    To keep response between each track from overlapping the -P/--pitch-jump and
    -T/--time-jump should be larger than all detector response.

    '''

    from wirecell.util.wires.array import endpoints_from_schema, correct_endpoint_array
    from wirecell.util.wires.persist import load as load_wires
    from wirecell.gen.morse import generate
    import numpy


    if output is None:
        raise click.BadParameter("no output file given")

    length = unitify(length)
    step_size = unitify(step_size)
    time_jump = unitify(time_jump)
    pitch_jump = int(pitch_jump)
    electron_density = unitify(electron_density)
    planes = list(map(int, planes.split(",")))

    wire_store = load_wires(wires)
    
    # Load all wire planes in given anode face and find absolute reference plane x.
    plane_wires = [correct_endpoint_array(endpoints_from_schema(wire_store, p, face, anode)) for p in (0,1,2)]

    # per depo charge
    nsteps = length/step_size
    charge = -1*abs(electron_density / step_size)

    colxs = plane_wires[2][:,:,0]
    refx = numpy.sum(colxs)/colxs.size + unitify(response)

    data, info = generate(plane_wires, refx, charge, length,
                          planes,
                          time_jump, pitch_jump, impact_jump)
    numpy.savez(output, depo_data_0=data, depo_info_0=info)
                      
@cli.command("morse-summary")
@click.option("--channel-ranges", default="0,800,1600,2560", # pdsp
              help="comma-separated list of channel idents defining ranges")
@click.option("-o",'--output', default=None,
              help="Give a file name to save summary in JSON format")
@click.argument("signal")
def morse_summary(channel_ranges, output, signal):
    '''
    Given a signal.npz of "morse" signals, find signal processing smearing.
    '''
    from wirecell.gen.morse import load_frame, frame_peaks, dump_frame_peaks

    channel_ranges = list(map(int,channel_ranges.split(",")))

    signal = load_frame(signal)
    pms = frame_peaks(signal, channel_ranges)
    dump_frame_peaks(output, pms)
        
@cli.command("morse-splat")
@click.option("-o",'--output', default="/dev/stdout",
              help="Give a file name to save summary")
@click.argument("summary")
def morse_splat(output, summary):
    '''
    Spit out configuration for DepoFluxSplat given a morse summary
    '''
    from .morse import load_frame_peaks
    summary = load_frame_peaks(summary)

    res = list()

    for pln in summary:
        if pln.total.sigma is None:
            raise ValueError(f'null total sigma in {pln.total}')
        tsigma = pln.total.sigma
        psigma = 0;
        chans = pln.chan
        ngood = 0
        for c in chans:
            if c.sigma:
                psigma += c.sigma
                ngood += 1
        psigma /= ngood
        res.append(dict(tick_sigma=tsigma, pitch_sigma=psigma))

    with open(output, "w") as out:
        if '.json' in output:
            # match schema for DepoFluxSplat
            dfs = dict(smear_long = [r['tick_sigma'] for r in res],
                       smear_tran = [r['pitch_sigma'] for r in res])
            out.write(json.dumps(dfs, indent=4))
        if '.org' in output:
            lines=[
                '|plane|tick|pitch|',
                '|-----|----|-----|'
            ]
            for l,r in zip("UVZ",res):
                line = f'|{l}|{r["tick_sigma"]:.3f}|{r["pitch_sigma"]:.3f}|'
                lines.append(line)
            print('\n'.join(lines))
            
        else:
            for one in res:
                out.write('sigmas: tick={tick_sigma} pitch={pitch_sigma}\n'.format(**one))


@cli.command("morse-plots")
@click.option("--channel-ranges", default="0,800,1600,2560", # pdsp
              help="comma-separated list of channel idents defining ranges")
@click.option("-t","--tick", default="0.5*us",
              help='The the sample period (def="0.5*us")')
@click.option("-o",'--output', default=None,
              help="Give a file name to save plots")
@click.option("-S","--summary", default=None,
              help='A morse summary JSON file.  Optional.  Will generate summary if omitted')
@click.argument("signal")
def morse_plots(channel_ranges, tick, output, summary, signal):
    '''
    Produce plots for analysis of "morse" track pattern.
    '''
    from wirecell.gen.morse import load_frame, frame_peaks, load_frame_peaks
    from wirecell.gen.plots.morse import width_plots

    tick = unitify(tick)
    channel_ranges = list(map(int,channel_ranges.split(",")))
    signal = load_frame(signal)
    if summary:
        fps = load_frame_peaks(summary)
    else:
        fps = frame_peaks(signal, channel_ranges)

    with pages(output) as out:
        width_plots(out, signal, fps, channel_ranges, tick)



@cli.command("depo-line")
@click.option("-S", "--step-size", default="1.0*mm",
              help="Distance between deposition of ionization electron groups")
@click.option("-T", "--time", default="0*ns",
              help="Time or uniform time range if two numbers over which the 't0' time for tracks are selected")
@click.option("-e", "--electron-density", default="5000/mm",
              help="Linear electron density on track (number of electrons per unit track length)")
@click.option("-f", "--first", type=str, default="0,0,0",
              help="The first track endpoint as 'x*unit,y*unit,z*unit' ")
@click.option("-l", "--last", type=str, default="0,0,0",
              help="The last track endpoint 'x*unit,y*unit,z*unit' ")
@click.option("-s", "--sigma", type=str, default="0,0",
              help="The longitudinal and transverse extent 'L,T'")
@click.option("--track-speed", default="clight",
              help="Speed of track")
@click.option("-o", "--output",
              type=click.Path(dir_okay=False, file_okay=True),
              help="Depo file which to save the results")
def depo_line(step_size, time, electron_density, first, last, sigma, track_speed, output):
    '''
    Generate a single line of depos between endpoints given in global coordinates.
    '''
    from .depogen import lines
    import numpy

    if output is None:
        raise click.BadParameter("no output file provided")

    step_size = unitify(step_size)
    time = unitify_parse(time)
    electron_density = unitify(electron_density)
    p0 = numpy.array(unitify_parse(first))
    p1 = numpy.array(unitify_parse(last))
    sigma = unitify_parse(sigma)
    track_speed = unitify(track_speed)

    eperstep = electron_density * step_size

    delt = p1 - p0
    dist = numpy.linalg.norm(delt)
    npts = 1 + int(round(dist/step_size))

    times = numpy.expand_dims(
        time + numpy.linspace(0, dist/track_speed, npts, endpoint=True), 1)
    charges = numpy.expand_dims(numpy.array((eperstep,)*npts), 1)
    points = numpy.linspace(p0, p1, npts, endpoint=True)
    sigmas = numpy.array(sigma * npts).reshape((-1,2))

    ids = numpy.expand_dims(numpy.arange(npts), 1)
    rest = numpy.array((0,0,0)*npts).reshape((-1,3))

    data = numpy.hstack((times,charges,points,sigmas), dtype='float32')
    info = numpy.hstack((ids,rest), dtype='int32')

    numpy.savez(output, depo_data_0=data, depo_info_0=info)


@cli.command("depo-lines")
@click.option("-e", "--electron-density", default="5000/mm",
              help="Linear electron density on track (number of electrons per unit track length)")
@click.option("-S", "--step-size", default="1.0*mm",
              help="Distance between deposition of ionization electron groups")
@click.option("-T", "--time", default="0*ns",
              help="Time or uniform time range if two numbers over which the 't0' time for tracks are selected")
@click.option("-t", "--tracks", default=1,
              help="Number of tracks per depo set")
@click.option("-s", "--sets", default=1,
              help="Number of depo sets")
@click.option("-c", "--corner", type=str, default="0,0,0",
              help="One corner of a bounding box")
@click.option("-d", "--diagonal", type=str, default="1*m,1*m,1*m",
              help="A vector from corner to diagonally opposed corner of bounding box")
@click.option("--track-speed", default="clight",
              help="Speed of track")
@click.option("--seed", default="0,1,2,3,4", type=str,
              help="A single integer or comma-list of integers to use as random seed")
@click.option("--track-info", default="array", 
              help="How to store meta data about the tracks: string 'array', 'omit' or a JSON file name")
@click.option("-o", "--output",
              type=click.Path(dir_okay=False, file_okay=True),
              help="Numpy file (.npz) in which to save the results")
def depo_lines(electron_density, step_size, time, tracks, sets,
               corner, diagonal, track_speed, seed, track_info, output):
    '''
    Generate ideal line-source "tracks" of depos
    '''
    import numpy.random
    import numpy

    if output is None or not output.endswith(".npz"):
        raise click.BadParameter(f'unsupported file type: {output}')

    seed = list(map(int, seed.split(",")))
    numpy.random.seed(seed)

    time = unitify_parse(time)
    track_speed = unitify(track_speed)
    electron_density = unitify(electron_density)
    step_size = unitify(step_size)
    eperstep = electron_density * step_size

    p0 = numpy.array(unitify_parse(corner))
    p1 = numpy.array(unitify_parse(diagonal)) + p0

    from .depogen import lines

    arrays = lines(tracks, sets, p0, p1, time, eperstep, step_size, track_speed)

    if track_info != "array":
        ti = list()
        for key in list(arrays.keys()):
            if key.startswith("track_info_"):
                arr = arrays.pop(key)
                dat = {name:arr[name].tolist() for name in arr.dtype.names}
                dat["ident"] = key[11:]
                ti.append(dat)
        if track_info != "omit":
            open(track_info, "w").write(json.dumps(ti))

    log.info("saving: %s" % str(list(arrays.keys())))
    numpy.savez(output, **arrays) 

@cli.command("depo-point")
@click.option("-n", "--number", default="5000",
              help="Number of electrons in the depo")
@click.option("-t", "--time", default="0*us",
              help="The time of the depo")
@click.option("-p", "--position", type=str, default="0,0,0",
              help="Position of the depo as 'x,y,z'")
@click.option("-s", "--sigma", type=str, default="0,0",
              help="The longitudinal and transverse extent 'L,T'")
@click.option("-o", "--output",
              type=click.Path(dir_okay=False, file_okay=True),
              help="Depo file which to save the results")
def depo_point(number, time, position, sigma, output):
    '''
    Generate a single point depo.
    '''
    import numpy

    if output is None:
        raise click.BadParameter("no output file provided")

    number = unitify(number)
    if number > 0:
        number = -number
    time = unitify(time)
    position = unitify_parse(position)
    sigma = unitify_parse(sigma)
    data = numpy.array([time,number]+position+sigma, dtype='float32').reshape(7,1)
    info = numpy.array((0, 0, 0, 0), dtype="int32").reshape(4,1)

    numpy.savez(output, depo_data_0=data, depo_info_0=info)


@cli.command("depo-sphere")
@click.option("-r", "--radius", default="1*m",
              help="Radius of the origin sphere)")
@click.option("-e", "--electron-density", default="5000/mm",
              help="Linear electron density on track (number of electrons per unit track length)")
@click.option("-S", "--step-size", default="1.0*mm",
              help="Distance between deposition of ionization electron groups")
@click.option("-c", "--corner", type=str, default="0,0,0",
              help="One corner of a bounding box")
@click.option("-d", "--diagonal", type=str, default="1*m,1*m,1*m",
              help="A vector from corner to diagonally opposed corner of bounding box")
@click.option("-O", "--origin", type=str, default="0,0,0",
              help="A vector to origin")
@click.option("--seed", default="0,1,2,3,4", type=str,
              help="A single integer or comma-list of integers to use as random seed")
@click.option("-o", "--output",
              type=click.Path(dir_okay=False, file_okay=True),
              help="Numpy file (.npz) in which to save the results")
def depo_sphere(radius, electron_density, step_size, 
                corner, diagonal, origin, seed, output):
    '''
    Generate ideal phere of depos
    '''
    import numpy

    if output is None:
        raise click.BadParameter("no output file provided")
    if not output.endswith(".npz"):
        raise click.BadParameter(f'unsupported file type: {output}')

    seed = list(map(int, seed.split(",")))
    import numpy.random
    numpy.random.seed(seed)

    radius = unitify_parse(radius)[0]
    electron_density = unitify(electron_density)
    step_size = unitify(step_size)
    eperstep = electron_density * step_size

    p0 = numpy.array(unitify_parse(corner))
    p1 = numpy.array(unitify_parse(diagonal))
    origin = numpy.array(unitify_parse(origin))

    from .depogen import sphere

    arrays = sphere(origin, p0, p1, radius=radius,
                    eperstep=eperstep, step_size=step_size)

    numpy.savez(output, **arrays) 


        

@cli.command("frame-stats")
@click.option("--plane-channels", default="800,800,960",
              help="comma list of channel counts per plane in u,v,w order")
@click.option("-o","--output", default="/dev/stdout",
              help="File to receive output")
@frame_input()
def frame_stats(array, plane_channels, ariofile, output, **kwds):
    '''
    Return (print) stats on the time distribution of a frame.

    '''
    import numpy

    def calc_stats(x):
        n = int(x.size)
        mu = numpy.mean(x)
        arel = numpy.abs(x-mu)
        rms = numpy.sqrt( (arel**2).mean() )
        outliers = [int(sum(arel >= sigma*rms)) for sigma in range(0,11)]
        return dict(n=n, mu=mu, rms=rms, outliers=outliers)

    channels = [int(c) for c in plane_channels.split(',')]
    chan0=0
    dat=dict()
    for chan, letter in zip(channels,"UVW"):
        plane = array[chan0:chan0+chan,:]
        plane = (plane.T - numpy.median(plane, axis=1)).T

        tsum = plane.sum(axis=0)/plane.shape[0]
        csum = plane.sum(axis=1)/plane.shape[1]

        log.debug(f'chans:{chan0}+{chan}')
        
        dat[letter] = dict(t=calc_stats(tsum), c=calc_stats(csum))
        # click.echo(' '.join([letter, 't'] + list(map(str,calc_stats(tsum)))))
        # click.echo(' '.join([letter, 'c'] + list(map(str,calc_stats(csum)))))
        chan0 += chan
    with open(output, "w") as fp:
        fp.write(json.dumps(dat, indent=4) + "\n")

# a decorator for common *linegen args
def linegen_args(func):

    @click.option(
        "-e", "--electron-density",
        default = "5000/mm",
        help    = (
            "Linear electron density on track"
            " (number of electrons per unit track length)"
        ),
        type    = str,
    )
    @click.option(
        "-S", "--step-size",
        default = "1.0*mm",
        help    = "Distance between deposition of ionization electron groups",
        type    = str,
    )
    @click.option(
        "-T", "--time",
        default = "0*ns",
        help    = "Start time of the track",
        type    = str,
    )
    @click.option(
        "--theta_y",
        default = "90*deg",
        help    = "Track angle with y axis of the coordinate system",
        type    = str,
    )
    @click.option(
        "--theta_xz",
        default = "45*deg",
        help    = "Track angle in the xz plane of the coordinate system",
        type    = str,
    )
    @click.option(
        "-l", "--length",
        default = "1*m",
        help    = "Track length",
        type    = str,
    )
    @click.option(
        "--track-speed", default = "clight", help = "Speed of track"
    )
    @click.option(
        "--angle-coords",
        default = "global",
        help    = (
            "Coordinate system in which angles theta_y and theta_xz are given"
        ),
        type    = click.Choice(['wire-plane', 'global'], case_sensitive = True),
    )
    @click.option(
        "-o", "--output_depos",
        type = click.Path(dir_okay = False, file_okay = True),
        help = "Path to save depo sets to",
        required = True,
    )
    @click.option(
        "-m", "--output_meta",
        type = click.Path(dir_okay = False, file_okay = True),
        help = "Path to save track metadata to",
    )
    @functools.wraps(func)
    def wrapper(*args, **kwds):
    
        # pylint: disable=too-many-arguments
        from .linegen import TrackConfig

        electron_density = unitify(kwds.pop("electron_density"))
        step_size        = unitify(kwds.pop("step_size"))

        kwds["track_config"] = TrackConfig(
            length        = unitify(kwds.pop("length")),
            t0            = unitify(kwds.pop("time")),
            eperstep      = electron_density * step_size,
            step_size     = step_size,
            theta_y       = unitify(kwds.pop("theta_y")),
            theta_xz      = unitify(kwds.pop("theta_xz")),
            track_speed   = unitify(kwds.pop("track_speed")),
            global_angles = (kwds.pop("angle_coords") == 'global')
        )
        return func(*args, **kwds)
    return wrapper


@cli.command("linegen")
@click.option(
    "--phi",
    default = ["60*deg"],
    help    = "Wire plane rotation, def=60*deg, multiple okay",
    type    = str,
    multiple = True
)
@click.option(
    "-c", "--center",
    default = "0*m,0*m,0*m",
    help    = "Track center",
    type    = str,
)
@linegen_args
def linegen(center, phi, track_config, output_depos, output_meta):
    '''
    Generate a line of depos given angles of wire direction.
    '''
    # pylint: disable=too-many-arguments
    from .linegen import generate_and_save_line_track

    center           = unitify(center)
    if len(phi) == 1:           # guess MB-like symmetry.
        phi.append("-1*"+phi[0])
    if len(phi) == 2:           # assume "usual" collection wire angle.
        phi.append("0")
    phi              = np.array([unitify(p) for p in phi])

    generate_and_save_line_track(
        center, track_config, phi, output_depos, output_meta
    )

@cli.command("detlinegen")
@click.option(
    "-d", "--detector",
    help = "Name of the detector",
    type = str,
    required = True,
)
@click.option(
    "--apa",
    default  = 0,
    help     = "APA index",
    type     = int,
    required = True,
)
@click.option(
    "--plane",
    default  = 0,
    help     = "Wire plane index",
    type     = int,
    required = True,
)
@click.option(
    "--offset",
    default = "-1*m,0*m,0*m",
    help    = "Track Offset from wire-plane center",
    type    = str,
)
@linegen_args
def detlinegen(detector, apa, plane, offset, track_config, output_depos, output_meta):
    '''
    Generate a line of depos given angles of direction for a specific detector.
    ''' 
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    from .linegen import generate_and_save_line_track_in_detector

    offset           = unitify(offset)

    generate_and_save_line_track_in_detector(
        detector, apa, plane, offset, track_config, output_depos, output_meta
    )


def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
    
    



