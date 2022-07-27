#!/usr/bin/env python3

import math
import click
from wirecell import units
from wirecell.util.functions import unitify, unitify_parse
from wirecell.util.cli import jsonnet_loader

cmddef = dict(context_settings = dict(help_option_names=['-h', '--help']))

@click.group("util", **cmddef)
@click.pass_context
def cli(ctx):
    '''
    Wire Cell Signal Simulation Commands
    '''

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
              help="The depo generation index")
@click.option("-i", "--index", default=0,
              help="The depos set index in the file")
@click.option("-p", "--plot", default='qxz',
              help="The plot to make.")
@click.option("-s", "--speed", default=None,
              help="Assign x position based on drift speed, use units like '1.6*mm/us'.")
@click.option("--t0", default="0*ns",
              help="Arbitrary additive time used in drift speed assignment, use units")
@click.argument("input-file")
@click.argument("output-file")
@click.pass_context
def plot_depos(ctx, generation, index, plot,
               speed, t0,
               input_file, output_file):
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

    plotter = getattr(deposmod, "plot_"+plot)
    depos = deposmod.load(input_file, index, generation)
    if 't' not in depos or len(depos['t']) == 0:
        print(f'No depos for index={index} and generation={generation} in {input_file}')
        return
    t0 = unitify(t0)
    if speed is not None:
        speed = unitify(speed)
        print(f'applying speed: {speed/(units.mm/units.us)} mm/us')
        depos['x'] = speed*(depos['t']+t0)
    else:
        depos['t'] += t0

    #depos = deposmod.remove_zero_steps(depos)
    try:
        plotter(depos, output_file)
    except ValueError as e:
        print(e)
        # punt, but make a file to satisfy workflow managers
        print(f'writing empty file {output_file}')
        import matplotlib.pyplot as plt
        plt.savefig(output_file)


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
    print (times)

    from wirecell.gen import sim
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    import numpy
    f = numpy.load(npz_file);

    fnums = [int(k.split('_')[-1]) for k in f.keys() if k.startswith("frame")]
    dnums = [int(k.split('_')[-1]) for k in f.keys() if k.startswith("depo_data")]

    with PdfPages(pdf_file) as pdf:

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
    import numpy
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from wirecell.util.plottools import NameSequence

    if output_file.endswith("pdf"):
        print(f'Saving to pdf: {output_file}')
        Outer = PdfPages
    else:
        print(f'Saving to: {output_file}')
        Outer = NameSequence

    if not time_range:
        if ticks:
            time_range = "0,-1"
        else:
            time_range = '0,5'

    
    fp = ario.load(input_file)

    numbers = [int(i.strip()) for i in number.split(",") if i.strip()]

    with Outer(output_file) as out:
        
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
                print ("Using channel groups: ", ch)

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
@click.option("-o", "--output",
              type=click.Path(dir_okay=False, file_okay=True),
              help="Numpy file (.npz) in which to save the results")
def depo_lines(electron_density, step_size, time, tracks, sets,
               corner, diagonal, track_speed, seed, output):
    '''
    Generate ideal line-source "tracks" of depos
    '''

    if not output.endswith(".npz"):
        print(f'unsupported file type: {output}')
        return -1

    seed = list(map(int, seed.split(",")))
    import numpy.random
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

    print("saving:", list(arrays.keys()))
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

    if not output.endswith(".npz"):
        print(f'unsupported file type: {output}')
        return -1

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



def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
    
    



