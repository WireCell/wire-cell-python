#!/usr/bin/env python3

import math
import click
from wirecell import units
from wirecell.util.functions import unitify, unitify_parse

@click.group("util")
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
@click.option("-j", "--json_path", default='depos',
                  help="Data structure path to the deposition array in the input file.")
@click.option("-p", "--plot", default='nxz',
                  help="The plot to make.")
@click.argument("input-file")
@click.argument("output-file")
@click.pass_context
def plot_depos(ctx, json_path, plot,
                   input_file, output_file):
    '''
    Make a plot from a WCT JSON depo file
    '''
    import depos as deposmod
    plotter = getattr(deposmod, "plot_"+plot)
    depos = deposmod.load(input_file)
    plotter(depos, output_file)

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
@click.pass_context
def plot_sim(ctx, input_file, output_file, ticks, plot, tag, time_range, number, channel_groups, channel_boundaries):
    '''
    Make plots of sim quantities saved into numpy array files.
    '''
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

    fp = numpy.load(input_file)

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
                out.savefig(fig)
                plt.close()

            if 'depo' in plot:

                deps = wirecell.gen.sim.Depos(fp, ident=onenum)
                fig, axes = deps.plot()
                out.savefig(fig)
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
@click.option("-c", "--diagonal", type=str, default="1*m,1*m,1*m",
              help="A vector from origin to diagonally opposed corner of bounding box")
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
    seed = list(map(int, seed.split(",")))
    import numpy.random
    numpy.random.seed(seed)

    import numpy
    from numpy.random import uniform, normal as gauss

    if not output.endswith(".npz"):
        print(f'unsupported file type: {output}')
        return -1

    time = unitify_parse(time)
    track_speed = unitify(track_speed)
    electron_density = unitify(electron_density)
    step_size = unitify(step_size)
    eperstep = electron_density * step_size

    p0 = numpy.array(unitify_parse(corner))
    p1 = numpy.array(unitify_parse(diagonal))
    bb = list(zip(p0, p1))
    pmid = 0.5 * (p0 + p1)

    tinfot = numpy.dtype([('pmin','3float32'), ('pmax','3float32'),
                          ('tmin', 'float32'), ('tmax', 'float32'),
                          ('step','f4'),       ('eper','f4')])

    collect = dict()
    for iset in range(sets):
        last_id = 0

        datas = list()
        infos = list()
        tinfos = numpy.zeros(tracks, dtype=tinfot)
        for itrack in range(tracks):

            pt = numpy.array([uniform(a,b) for a,b in bb])
            g3 = numpy.array([gauss(0, 1) for i in range(3)])
            mag = math.sqrt(numpy.dot(g3,g3))
            vdir = g3/mag
            
            t0 = (p0 - pt ) / vdir # may have zeros
            t1 = (p1 - pt ) / vdir # may have zeros
            
            a0 = numpy.argmin(numpy.abs(t0))
            a1 = numpy.argmin(numpy.abs(t1))

            # points on either side bb walls
            pmin = pt + t0[a0] * vdir
            pmax = pt + t1[a1] * vdir

            dp = pmax - pmin
            pdist = math.sqrt(numpy.dot(dp, dp))
            nsteps = int(round(pdist / step_size))

            pts = numpy.linspace(pmin,pmax, nsteps+1, endpoint=True)
            
            if len(time) == 1:
                time0 = time[0]
            else:
                time0 = uniform(time[0], time[1])

            timef = nsteps*step_size/track_speed
            times = numpy.linspace(time0, timef, nsteps+1, endpoint=True)

            dt = timef-time0
            print(f'nsteps:{nsteps}, pdist:{pdist/units.mm:.1f} mm, dt={dt/units.ns:.1f} ns, {eperstep}')

            tinfos["pmin"][itrack] = pmin
            tinfos["pmax"][itrack] = pmax
            tinfos["tmin"][itrack] = time0
            tinfos["tmax"][itrack] = timef         
            tinfos["step"][itrack] = step_size
            tinfos["eper"][itrack] = eperstep

            charges = numpy.zeros(nsteps+1) + eperstep

            zeros = numpy.zeros(nsteps+1)

            data = numpy.vstack([
                times,
                charges,
                pts.T,
                zeros,
                zeros])

            ids = numpy.arange(last_id, last_id + nsteps + 1)
            last_id = ids[-1]

            info = numpy.vstack([
                ids,
                zeros,
                zeros,
                zeros])

            datas.append(data)
            infos.append(info)

        datas = numpy.vstack([d.T for d in datas])
        infos = numpy.vstack([i.T for i in infos])

        # datas is now as (n,7)

        timeorder = numpy.argsort(datas[:,0])
        datas = datas[timeorder]
        infos = infos[timeorder]

        collect[f'depo_data_{iset}'] = numpy.array(datas, dtype='float32')
        collect[f'depo_info_{iset}'] = numpy.array(infos, dtype='int32')
        collect[f'track_info_{iset}'] = tinfos

    # fixme: nice to add support for bee and wct JSON depo files
    print("saving:", list(collect.keys()))
    numpy.savez(output, **collect) 

def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
    
    



