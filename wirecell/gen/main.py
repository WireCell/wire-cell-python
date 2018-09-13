import click

from wirecell import units

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
@click.option("-n", "--number", default=0,
                  help="The number of the frame or depo set to plot.")
@click.option("-c", "--channel-groups", default='',
                  help="Indices of channel groups as comma separated list.")
@click.option("-b", "--channel-boundaries", default='',
                  help="Channels at which there are boundaries.")
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

    if not time_range:
        if ticks:
            time_range = "0,-1"
        else:
            time_range = '0,5'

    fp = numpy.load(input_file)

    if 'frame' in plot:
        print "Frames: %s" %(', '.join([k for k in fp.keys() if k.startswith("frame")]), )
        fr = wirecell.gen.sim.Frame(fp, tag=tag, ident=number)
        
        channel_boundaries = wirecell.gen.sim.parse_channel_boundaries(channel_boundaries)
        ch = wirecell.gen.sim.group_channel_indices(fr.channels, channel_boundaries)
        print "All channel groups: ", ch
        if channel_groups:
            ch = [ch[int(ci)] for ci in channel_groups.split(",")]
        print "Using groups: ", ch
        

        if ticks:
            plotter = fr.plot_ticks
            t0,tf = [int(t,10) for t in time_range.split(",")]
        else:
            plotter = fr.plot
            t0,tf = [float(t)*units.ms for t in time_range.split(",")]


        fig, axes = plotter(t0, tf, raw=False, chinds=ch)
        plt.savefig(output_file)

    if 'depo' in plot:
        print "Depos: %s" %(', '.join([k for k in fp.keys() if k.startswith("depo_data")]), )
        deps = wirecell.gen.sim.Depos(fp, ident=number)
        fig, axes = deps.plot()
        plt.savefig(output_file)

def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
    
    



