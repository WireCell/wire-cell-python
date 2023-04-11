#!/usr/bin/env python
'''
The wirecell-img main
'''
import os
import sys
import json
import click
import pathlib
from collections import Counter
import numpy
import matplotlib.pyplot as plt
from wirecell import units
from wirecell.util.functions import unitify, unitify_parse
from wirecell.util import ario
from wirecell.util.plottools import pages

import functools
import wirecell.gen.depos as deposmod
from . import tap, converter

from scipy.spatial.transform import Rotation
from zipfile import ZipFile
from zipfile import ZIP_DEFLATED as ZIP_COMPRESSION
### bzip2 is actually worse than deflate for depos!
# from zipfile import ZIP_BZIP2 as ZIP_COMPRESSION
from io import BytesIO

cmddef = dict(context_settings = dict(help_option_names=['-h', '--help']))

@click.group("img", **cmddef)
@click.pass_context
def cli(ctx):
    '''
    Wire Cell Toolkit Imaging Commands

    A cluster file is produced by ClusterFileSink and is an archive
    holding JSON or Numpy or as a special case may be a single JSON.

    '''
    pass


# 1. wrapper to handle undrift and loading of depo and cluster files.
# 2. retrofit all the commands.

def cluster_file(func):
    ''' A CLI decorator giving the command a "clusters" argument
    providing a generator of cluster graphs.  '''

    @click.option("-B", "--undrift-blobs", type=str, default=None,
                  help="Undrift with '<speed>,<time>', eg '1.6*mm/us,314*us'")
    @click.argument("cluster-file")
    @functools.wraps(func)
    def wrapper(*args, **kwds):
        cf = kwds.pop("cluster_file")
        cgraphs = tap.load(cf)
        ub = kwds.pop("undrift_blobs")
        if ub is not None:
            speed, time = unitify_parse(ub)
            cgraphs = [converter.undrift_blobs(cg, speed, time) for cg in cgraphs]
        kwds['clusters'] = cgraphs
        return func(*args, **kwds)
    return wrapper


def deposet_file(func):
    '''A CLI decorator giving the command a "depos" argument
    providing a generator-of-dict-of-array of depos.'''
    @click.option("-D", "--undrift-depos", type=str, default=None,
                  help="Undrift with '<speed>,<time>', eg '1.6*mm/us,314*us'")
    @click.option("-g", "--generation", default=0,
                  help="The depo generation index")
    @click.argument("depo-file")
    @functools.wraps(func)
    def wrapper(*args, **kwds):
        df = kwds.pop("depo_file")
        gen = kwds.pop("generation")
        deposets = deposmod.stream(df, gen)

        ud = kwds.pop("undrift_depos")
        if ud is not None:
            speed, time = unitify_parse(ud)
            deposets = [converter.undrift_depos(depos, speed, time) for depos in deposets]
        kwds['deposets'] = deposets
        return func(*args, **kwds)
    return wrapper
    
def paraview_file(ext, percent="-%03d"):
    '''A CLI decorator for a paraview_file argument.

    The "ext" should include the dot like ".vtp".

    The "percent" if defined is what is inserted between file base and
    extension names.
    '''
    def decorator(func):
        @click.argument("paraview-file")
        @functools.wraps(func)
        def wrapper(*args, **kwds):
            pf = kwds.pop("paraview_file")
            
            if not pf.endswith(ext):
                print (f'paraview expects a {ext} file extension, fixing')
                b = os.path.splitext(pf)[0]
                pf = b + ext

            if percent and '%' not in pf:
                print(f"Warning: no '%d' code found in {pf}, will add one")
                b,e = os.path.splitext(pf)
                pf = b + percent + e
            kwds['paraview_file'] = pf
            return func(*args, **kwds)
        return wrapper
    return decorator


import wirecell.img.plot_depos_blobs as depo_blob_plotters
depo_blob_plots = [name[5:] for name in dir(depo_blob_plotters) if name.startswith("plot_")]
@cli.command("plot-depos-blobs")
@click.option("-i", "--index", default=None,
              help="The file index, default is multipage")
@click.option("-p", "--plot", default='x',
              type=click.Choice(depo_blob_plots),
              help="The plot to make.")
@deposet_file
@cluster_file
@click.argument("plot-file")
@click.argument("params", nargs=-1)
def plot_depos_blobs(index, plot, deposets, clusters, plot_file, params):
    '''Plots combining depos and blobs.'''

    if index is not None:
        deposets = [list(deposets)[index]]
        clusters = [list(clusters)[index]]

    kwargs = dict()
    for p in params:
        k,v = p.split("=",1)
        v = unitify(v)
        kwargs[k] = v
        print(f'{k} : {v}')

    plotter = getattr(depo_blob_plotters, "plot_"+plot)

    with pages(plot_file) as printer:

        for count, (depos, cgraph) in enumerate(zip(deposets, clusters)):
                    
            if 0 == cgraph.number_of_nodes():
                if index is not None:
                    count = index
                click.echo(f'empty cluster at index={count} of file cluster_file')
                return

            fig = plotter(depos, cgraph, **kwargs)
            printer.savefig(dpi=300)
            break               # fixme
    click.echo(plot_file)


import wirecell.img.plot_blobs as blob_plotters
blob_plots = [name[5:] for name in dir(blob_plotters) if name.startswith("plot_")]

@cli.command("plot-blobs")
@click.option("-p", "--plot", default='x',
              type=click.Choice(blob_plots),
              help="The plot to make.")
@cluster_file
@click.argument("plot-file")
@click.pass_context
def plot_blobs(ctx, plot, clusters, plot_file):
    '''
    Produce plots related to blobs in cluster.
    '''
    from . import tap, converter
    plotter = getattr(blob_plotters, "plot_"+plot)

    with pages(plot_file) as printer:

        for gr in clusters:
            nnodes = gr.number_of_nodes()

            if 0 == nnodes:
                click.echo("no verticies")
                return

            try:
                fig = plotter(gr)
            except ValueError:
                print(f'failed to plot graph with {nnodes} vertices')
                continue
            printer.savefig()
    click.echo(plot_file)

@cli.command("dump-blobs")
@cluster_file
@click.option("-o", "--output", default="/dev/stdout", help="output file name")
@click.option("-s", "--signals", default=None, help="file to dump signals")
@click.pass_context
def dump_blobs(ctx, clusters, output, signals):
    '''
    dump blob signatures in cluster to a file.
    '''
    import wirecell.img.dump_blobs as db
    for gr in clusters:
        db.dump_blobs(gr, signals, output)

@cli.command("dump-bb-clusters")
@cluster_file
@click.pass_context
def dump_bb_clusters(ctx, clusters):
    '''
    dump blob cluster signitures
    '''
    import wirecell.img.dump_bb_clusters as dc
    for gr in clusters:
        dc.dump_bb_clusters(gr)


@cli.command("inspect")
@click.argument("cluster-file")
@click.pass_context
def inspect(ctx, cluster_file):
    '''
    Inspect a cluster file
    '''
    from . import converter, tap, clusters

    path = pathlib.Path(cluster_file)
    if not path.exists():
        print(f'no such file: {path}')
        return

    if path.name.endswith(".json"):
        print ('JSON file assuming from JsonClusterTap')
    elif '.tar' in path.name:
        print ('TAR file assuming from ClusterFileSink')

    graphs = list(tap.load(str(path)))
    print (f'number of graphs: {len(graphs)}')
    for ig, gr in enumerate(graphs):
        cm = clusters.ClusterMap(gr)

        print(f'{ig}: {gr.number_of_nodes()} vertices, {gr.number_of_edges()} edges')
        counter = Counter(dict(gr.nodes(data='code')).values())
        for code, count in sorted(counter.items()):
            print(f'\t{code}: {count} nodes')

            if code == 'b':
                q = sum([n['value'] for c,n in gr.nodes(data=True) if n['code'] == code])
                print(f'\t\ttotal charge: {q}')
                continue

            if code == 's':
                q=0
                for snode in cm.nodes_oftype('s'):
                    sdat = cm.gr.nodes[snode]
                    sig = sdat['signal']
                    q += sum([v['val'] for v in sig.values()])
                print(f'\t\ttotal charge: {q}')
                continue
        

@cli.command("paraview-blobs")
@cluster_file
@paraview_file(".vtu")
@click.pass_context
def paraview_blobs(ctx, clusters, paraview_file):
    '''
    Convert a cluster file to a ParaView .vtu files of blobs

    Speed and t0 converts time to relative drift coordinate.
    '''
    from . import converter, tap
    from tvtk.api import write_data

    if len(clusters) == 0:
        print('no graphs')
        sys.exit(-1)

    for n, gr in enumerate(clusters):
        if 0 == gr.number_of_nodes():
            click.echo("no verticies")
            sys.exit(-1)
        dat = converter.clusters2blobs(gr)
        fname = paraview_file
        if '%' in paraview_file:
            fname = paraview_file%n
        write_data(dat, fname)
        click.echo(fname)

    return


@cli.command("paraview-activity")
@cluster_file
@paraview_file(".vti")
@click.pass_context
def paraview_activity(ctx, clusters, paraview_file):
    '''
    Convert cluster files to ParaView .vti files of activity
    '''
    from . import converter, tap
    from tvtk.api import write_data
    
    for n, gr in enumerate(clusters):
        if 0 == gr.number_of_nodes():
            click.echo("no verticies")
            return

        fname,ext=os.path.splitext(paraview_file)
        if '%' in fname:
            fname = fname%n

        alldat = converter.clusters2views(gr)
        for wpid, dat in alldat.items():
            pname = f'{fname}-plane{wpid}{ext}'
            write_data(dat, pname)
            click.echo(pname)

    return


@cli.command("paraview-depos")
@click.option("-i", "--index", default=None,
              help="The depos set index in the file")
@deposet_file
@paraview_file(".vtp")
@click.pass_context
def paraview_depos(ctx, index, deposets, paraview_file):
    '''Convert WCT depo file a ParaView .vtp file.

    If index is given, the single deposet is converted to the output
    file.  If no index then each deposet will be converted to a file
    and the paraview-file argument should contain a %d to receive a
    count.

    See also "wirecell-gen plot-depos".

    '''
    from . import converter
    from tvtk.api import write_data

    for count, deposet in enumerate(deposets):
        if index is not None and index != count:
            continue

        ugrid = converter.depos2pts(deposet)

        fname = paraview_file
        if '%' in fname:
            fname = fname % count

        write_data(ugrid, fname)
        click.echo(fname)
        return



#    Bee support:   
#    http://bnlif.github.io/wire-cell-docs/viz/uploads/


@cli.command("bee-blobs")
@click.option('-o', '--output', help="The output Bee JSON file name")
@click.option('-g', '--geom', default="protodune",
              help="The name of the detector geometry")
@click.option('--rse', nargs=3, type=int, default=[0, 0, 0],
              help="The '<run> <subrun> <event>' numbers as a triple of integers")
@click.option('-s', '--sampling', type=click.Choice(["center","uniform"]), default="uniform",
              help="The sampling technique to turn blob volumes into points")
@click.option("--speed", default="1.6*mm/us",
              help="Drift speed (with units)")
@click.option("--t0", default="0*ns",
              help="Absolute time of first tick (with units)")
@click.option('-d', '--density', type=float, default=9.0,
              help="For samplings which care, specify target points per cc")
@click.argument("cluster-files", nargs=-1)
def bee_blobs(output, geom, rse, sampling, speed, t0, density, cluster_files):
    '''
    Produce a Bee JSON file from a cluster file.
    '''
    from . import tap, converter

    speed = unitify(speed)
    t0 = unitify(t0)

    dat = dict(runNo=rse[0], subRunNo=rse[1], eventNo=rse[2], geom=geom, type="cluster",
               x=list(), y=list(), z=list(), q=list(), cluster_id=list())

    def fclean(arr):
        return [round(a, 3) for a in arr]
        
    # given by user in units of 1/cc.  Convert to system of units 1/L^3.
    density *= 1.0/(units.cm**3)
    sampling_func = dict(
        center = converter.blob_center,
        uniform = lambda b : converter.blob_uniform_sample(b, density),
    )[sampling]

    import networkx as nx
    def nodes_oftype(gr, typecode):
        return [n for n,d in gr.nodes.data() if d['code'] == typecode]

    for ctf in cluster_files:
        gr = list(tap.load(ctf))[0] # fixme: for now ignore subsequent graphs
        gr = converter.undrift_blobs(gr, speed, t0)
        print ("got %d" % gr.number_of_nodes())
        if 0 == gr.number_of_nodes():
            print("skipping empty graph %s" % ctf)
            continue
        bnodes = nodes_oftype(gr,'b')
        bgr = gr.subgraph(bnodes)
        cluster_id = 0
        for bc in nx.connected_components(bgr):
            arr = converter.blobpoints(gr.subgraph(bc), sampling_func)
            if len(arr.shape) < 2:
                continue
            print ("%s: %d points" % (ctf, arr.shape[0]))
            dat['x'] += fclean(arr[:,0]/units.cm)
            dat['y'] += fclean(arr[:,1]/units.cm)
            dat['z'] += fclean(arr[:,2]/units.cm)
            dat['q'] += fclean(arr[:,3])
            dat['cluster_id'] += [cluster_id for i in range(len(arr[:,0]))]
            cluster_id += 1

    import json
    # monkey patch
    from json import encoder
    encoder.FLOAT_REPR = lambda o: format(o, '.3f')
    json.dump(dat, open(output,'w', encoding="utf8"))


def divine_planes(nch):
    '''
    Return list of channels in each plane based on total.
    '''
    if nch == 2560:             # protodune
        return [400, 400, 400, 400, 480, 480]
    if nch == 8256:             # microboone
        return [2400, 2400, 3456]
    print(f'not a canonical number of channels in a known detector: {nch}')
    return [nch]

@cli.command("activity")
@click.option('-o', '--output', help="The output plot file name")
@click.option('-s', '--slices', nargs=2, type=int, 
              help="Range of slice IDs")
@click.option('-S', '--slice-line', type=int, default=-1,
              help="Draw a line down a slice")
@click.option("--speed", default="1.6*mm/us",
              help="Drift speed (with units)")
@click.option("--t0", default="0*ns",
              help="Absolute time of first tick (with units)")
@click.argument("cluster-file")
def activity(output, slices, slice_line, speed, t0, cluster_file):
    '''
    Plot activity
    '''
    from matplotlib.colors import LogNorm
    from . import tap, clusters, plots

    speed = unitify(speed)
    t0 = unitify(t0)

    gr = list(tap.load(cluster_file))[0]
    gr = converter.undrift_blobs(gr, speed, t0)
    cm = clusters.ClusterMap(gr)
    ahist = plots.activity(cm)
    arr = ahist.arr
    print(f'channel x slice array shape: {arr.shape}')
    extent = list()
    if slices:
        arr = arr[:,slices[0]:slices[1]]
        extent = [slices[0], slices[1]]
    else:
        extent = [0, arr.shape[1]]
    extent += [ahist.rangey[1], ahist.rangey[0]]

    fig,ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(8.5,11.0)

    cmap = plt.get_cmap('gist_rainbow')
    im = ax.imshow(arr, cmap=cmap, interpolation='none', norm=LogNorm(), extent=extent)
    if slice_line > 0:
        ax.plot([slice_line, slice_line], [ahist.rangey[0], ahist.rangey[1]],
                linewidth=0.1, color='black')

    boundary = 0
    for chunk in divine_planes(arr.shape[0]):
        boundary += chunk
        y = boundary + ahist.rangey[0]
        ax.plot(extent[:2], [y,y], color='gray', linewidth=0.1);

    from matplotlib.ticker import  AutoMinorLocator
    minorLocator = AutoMinorLocator()
    ax.yaxis.set_minor_locator(minorLocator)
    ax.tick_params(which="both", width=1)
    ax.tick_params(which="major", length=7)
    ax.tick_params(which="minor", length=3)

    try:
        plt.colorbar(im, ax=ax)
    except ValueError:
        print("colorbar complains, probably have zero data")
        print('total:', numpy.sum(arr))
        return
        pass
    ax.set_title(cluster_file)
    ax.set_xlabel("slice ID")
    ax.set_ylabel("channel IDs")
    fig.savefig(output)


@cli.command("blob-activity-mask")
@click.option('-o', '--output', help="The output plot file name")
@click.option('-s', '--slices', nargs=2, type=int, 
              help="The output plot file name")
@click.option('-S', '--slice-line', type=int, default=-1,
              help="Draw a line down a slice")
@click.option("--speed", default="1.6*mm/us",
              help="Drift speed (with units)")
@click.option("--t0", default="0*ns",
              help="Absolute time of first tick (with units)")
@click.option('--found/--missed', default=True,
              help="Mask what blobs found or missed")
@click.argument("cluster-file")
def blob_activity_mask(output, slices, slice_line, speed, t0, found, cluster_file):
    '''
    Plot blobs as maskes on channel activity.
    '''
    from . import tap, clusters, plots

    speed = unitify(speed)
    t0 = unitify(t0)

    gr = list(tap.load(cluster_file))[0] # fixme
    gr = converter.undrift_blobs(gr, speed, t0)
    cm = clusters.ClusterMap(gr)
    ahist = plots.activity(cm)
    bhist = ahist.like()
    plots.blobs(cm, bhist)
    if found:
        sel = lambda a: a>= 1
        title="found mask"
    else:
        sel = lambda a: a < 1
        title="missed mask"
    extent = list()
    if slices:
        a = ahist.arr[:,slices[0]:slices[1]]
        b = bhist.arr[:,slices[0]:slices[1]]
        extent = [slices[0], slices[1]]
    else:
        a = ahist.arr
        b = bhist.arr
        extent = [0, a.shape[1]]
    extent += [ahist.rangey[1], ahist.rangey[0]]

    fig,ax = plots.mask_blobs(a, b, sel, extent)
    if slice_line > 0:
        ax.plot([slice_line, slice_line], [ahist.rangey[0], ahist.rangey[1]],
                linewidth=0.1, color='black')
    ax.set_title("%s %s" % (title, cluster_file))
    ax.set_xlabel("slice ID")
    ax.set_ylabel("channel IDs")
    fig.savefig(output)


@cli.command("wire-slice-activity")
@click.option('-o', '--output', help="The output plot file name")
@click.option('-s', '--sliceid', type=int, help="The slice ID to plot")
@click.option("--speed", default="1.6*mm/us",
              help="Drift speed (with units)")
@click.option("--t0", default="0*ns",
              help="Absolute time of first tick (with units)")
@click.argument("cluster-file")
def wire_slice_activity(output, sliceid, speed, t0, cluster_file):
    '''
    Plot the activity in one slice as wires and blobs
    '''
    from . import tap, clusters, plots
    speed = unitify(speed)
    t0 = unitify(t0)
    gr = next(tap.load(cluster_file))
    gr = converter.undrift_blobs(gr, speed, t0)
    cm = clusters.ClusterMap(gr)
    fig, axes = plots.wire_blob_slice(cm, sliceid)
    fig.savefig(output)


@cli.command("anidfg")
@click.option("-o", "--output", default="anidfg.gif", help="Output file")
@click.argument("logfile")
def anidfg(output, logfile):
    '''
    Produce an animated graph visualization from a log produced by
    TbbFlow with "dfg" output.
    '''
    from . import anidfg
    log = anidfg.parse_log(open(logfile))
    ga = anidfg.generate_graph(log)
    anidfg.render_graph(ga, output)


@cli.command("transform-depos")
@click.option("--forward/--no-forward", default=False,
              help='Forward the input array to output prior to transformed') 
@click.option("-l", "--locate", type=str, default=None,
              help='Locate center-of-charge to given position')
@click.option("-m", "--move", type=str, default=None,
              help='Translate by a relative displacement vector')
@click.option("-r", "--rotate", type=str, default=None, multiple=True,
              help='Rotate by given angle along each axis')
@click.option("-o", "--output", default=None,
              help="Send results to output (def: stdout)")
@click.argument("depos")
def transform_depos(forward, locate, move, rotate, output, depos):
    '''
    Apply zero or more transformations one or more times to
    distributions of depos.

    Rotations are applied about the center of charge of the
    distribution.  Multiple rotations and of different types may be
    given.  The type of rotation is given by a code letter followed by
    a colon followed by arguments.  When values are required they are
    provided as comma-separated list of values with units.  Rotation
    codes are:

        - q: quaternarian expecting 4 angles

        - x: single Euler angle about x axis

        - y: single Euler angle about y axis

        - z: single Euler angle about z axis

        - v: rotation vector, direction is axis, norm is angle

    Example rotation (the two would cancel each other)

    --rotate 'x:90*deg' --rotate 'v:-pi/2,0,0'

    A locate is applied prior to a move.  Both are given as a
    comma-spearated list of coordinates with units.  Eg to center on
    origin and offset

    --locate '0,0,0' --move '1*m,2*cm,3*mm'
    '''

    fp = ario.load(depos)
    indices = list()
    for k in fp:
        if k.startswith("depo_data_"):
            ind = int(k.split("_")[2])
            indices.append(ind)
    indices.sort();

    out_count = 0
    with ZipFile(output or "/dev/stdout", "w", compression=ZIP_COMPRESSION) as zf:

        def save(name, arr):
            print(name, arr.shape)
            bio = BytesIO()
            numpy.save(bio, arr)
            bio.seek(0)
            zf.writestr(name, bio.getvalue())
            
        generation = 0              # fixme, make configurable
        for index in indices:
            dat = fp[f'depo_data_{index}']
            nfo = fp[f'depo_info_{index}']

            if dat.shape[0] == 7:
                dat = dat.T
            if nfo.shape[0] == 4:
                nfo = nfo.T

            keep = nfo[:,2] == generation
            nfo = nfo[keep,:]
            dat = dat[keep,:]

            if forward:
                save(f'depo_data_{out_count}.npy', dat)
                save(f'depo_info_{out_count}.npy', nfo)
                out_count += 1

            q = dat[:,1]
            r = dat[:,2:5]

            qr = (q*r.T).T
            coq = numpy.sum(qr, axis=0)/numpy.sum(q)

            for one in rotate:
                code, args = one.split(":",1)
                args = numpy.array(unitify(args.split(",")))
                print(f'rotate {code} {args}')
                if code == 'q':
                    R = Rotation.from_quat(args)
                elif code == 'v':
                    R = Rotation.from_rotvec(args)
                else:
                    R = Rotation.from_euler(code, args[0])

                r = R.apply(r-coq)+coq

            if locate:
                locate = numpy.array(unitify(locate))
                r = r - coq + locate

            if move:
                move = numpy.array(unitify(move))
                r = r + move

            dat[:,2:5] = r
            save(f'depo_data_{out_count}.npy', dat)
            save(f'depo_info_{out_count}.npy', nfo)
            out_count += 1


def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
    
