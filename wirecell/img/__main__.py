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
from wirecell.util.cli import log, context, image_output

import functools
import wirecell.gen.depos as deposmod
from . import tap, converter

from scipy.spatial.transform import Rotation
from zipfile import ZipFile
from zipfile import ZIP_DEFLATED as ZIP_COMPRESSION
### bzip2 is actually worse than deflate for depos!
# from zipfile import ZIP_BZIP2 as ZIP_COMPRESSION
from io import BytesIO


@context("img")
def cli(ctx):
    """
    Wire-Cell Toolkit commands related to imaging.
    """
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
                log.warning (f'paraview expects a {ext} file extension, fixing')
                b = os.path.splitext(pf)[0]
                pf = b + ext

            if percent and '%' not in pf:
                log.warning(f"no '%d' code found in {pf}, will add one")
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
        log.debug(f'{k} : {v}')

    plotter = getattr(depo_blob_plotters, "plot_"+plot)

    with pages(plot_file) as printer:

        for count, (depos, cgraph) in enumerate(zip(deposets, clusters)):
                    
            if 0 == cgraph.number_of_nodes():
                if index is not None:
                    count = index
                log.warning(f'empty cluster at index={count} of file cluster_file')
                return

            fig = plotter(depos, cgraph, **kwargs)
            printer.savefig(dpi=300)
            break               # fixme
    log.info(plot_file)


import wirecell.img.plot_blobs as blob_plotters
blob_plots = [name[5:] for name in dir(blob_plotters) if name.startswith("plot_")]

@cli.command("plot-blobs")
@click.option("-p", "--plot", default='x',
              type=click.Choice(blob_plots),
              help="The plot to make.")
@cluster_file
@image_output
@click.pass_context
def plot_blobs(ctx, plot, clusters, output, **kwds):
    '''
    Produce plots related to blobs in cluster.
    '''
    from . import tap, converter
    plotter = getattr(blob_plotters, "plot_"+plot)

    with output as printer:

        for gr in clusters:
            nnodes = gr.number_of_nodes()

            if 0 == nnodes:
                log.error("no verticies")
                return

            try:
                fig = plotter(gr)
            except ValueError:
                log.error(f'failed to plot graph with {nnodes} nodes')
                continue
            printer.savefig()


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
    dump blob cluster signatures
    '''
    import wirecell.img.dump_bb_clusters as dc
    for gr in clusters:
        dc.dump_bb_clusters(gr)


@cli.command("inspect")
@click.option("-o", "--output", default="/dev/stdout", help="output file name")
@click.option("--verbose", default=False, is_flag=True, help="output a lot more")
@click.argument("cluster-file")
@click.pass_context
def inspect(ctx, output, verbose, cluster_file):
    '''
    Inspect a cluster file
    '''
    from . import converter, tap, clusters

    out = open(output,"w")

    def out_stats(name, dat):
        ntot = len(dat)
        zeros = [d for d in dat if d == 0]
        dat = [d for d in dat if d > 0]
        vtot = sum(dat)
        vmean = vmin = vmax = None
        n = len(dat)
        nz = ntot - n
        if n > 0:
            vmean = vtot / ntot
            vmin = min(dat)
            vmax = max(dat)

        out.write(f'\t\tmean {name} in {ntot} ({nz} zeros): {vmin} <= {vmean} <= {vmax}, tot:{vtot}\n')

    path = pathlib.Path(cluster_file)
    if not path.exists():
        log.error(f'no such file: {path}')
        return

    graphs = list(tap.load(str(path)))
    out.write(f'number of graphs: {len(graphs)}\n')
    for ig, gr in enumerate(graphs):
        # def by_code(code):
        #     return [n for c,n in gr.nodes(data=True) if n['code'] == code]

        cm = clusters.ClusterMap(gr)

        out.write(f'{ig}: {gr.number_of_nodes()} nodes\n')
        node_counter = Counter(dict(gr.nodes(data='code')).values())
        for code, count in sorted(node_counter.items()):

            ndat = cm.data_oftype(code)
            nodes_of_type = cm.nodes_oftype(code)

            keys = set()
            for n in nodes_of_type:
                keys.update(gr.nodes[n].keys())
            keys = list(keys)
            keys.sort()

            uniq_idents = set()
            uniq_descs = set()
            neighbor_counts = Counter()
            for n in nodes_of_type:
                uniq_idents.add(gr.nodes[n]['ident'])
                uniq_descs.add(gr.nodes[n]['desc'])
                for nn in gr[n]:
                    neighbor_counts[gr.nodes[nn]['code']] += 1
            out.write(f'\t{code}: {count} nodes ({len(uniq_idents)} idents, {len(uniq_descs)} descs), neighbors:')
            for c,n in sorted(neighbor_counts.items()):
                out.write(f' {c}={n}')
            out.write(f' data: {keys}\n')
            if code != 'w':
                assert len(uniq_idents) == count
            assert len(uniq_descs) == count
            def burp_neighbors():
                for n in nodes_of_type:
                    d = gr.nodes[n]
                    ident = d['ident']
                    out.write(f'\t\tnn for {code} {ident}:')
                    nnc = Counter()
                    for nn in gr[n]:
                        nnc[gr.nodes[nn]['code']] += 1
                    for c,n in sorted(nnc.items()):
                        out.write(f' {c}={n}')
                    out.write('\n')

            if code == 'b':
                for key in ['val', 'unc']:
                    out_stats(key, [n[key] for n in ndat])
                if verbose:
                    burp_neighbors()
                continue

            if code == 'c':
                for key in ['val', 'unc']:
                    # we do a get because some channels are not
                    # reachable from the slice->blob->measure that is
                    # done to add 'val' key.
                    out_stats(key, [n.get(key, 0) for n in ndat])
                continue

            if code == 'w':
                for thing in ['seg', 'wpid']:
                    c = Counter([n.get(thing,-1) for n in ndat])
                    out.write(f'\t\t{thing}: {c}\n')

            if code == 'm':
                for key in ['val', 'unc']:
                    out_stats(key, [n[key] for n in ndat])
                if verbose:
                    burp_neighbors()
                continue

            if code == 's':
                sigs = list()
                errs = list()
                nums = list()
                for snode in nodes_of_type:
                    sdat = cm.gr.nodes[snode]
                    ident = sdat['ident']
                    sig = sdat['signal']
                    sval = [v['val'] for v in sig]
                    serr = [v['unc'] for v in sig]
                    if verbose:
                        out_stats(f"sunc{ident}", serr)
                        out_stats(f"sval{ident}", sval)
                    nums.append(len(sval))
                    sigs += sval
                    errs += serr

                out_stats("val", sigs)
                out_stats("unc", errs)
                out_stats("num", nums)
                continue

        out.write(f'{ig}: {gr.number_of_edges()} edges\n')
        # A truly unholly counter
        edge_counter = Counter([''.join(sorted([gr.nodes[e[0]]["code"],gr.nodes[e[1]]["code"]])) for e in gr.edges])
        for code, count in sorted(edge_counter.items()):
            out.write(f'\t{code}: {count} edges\n')



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
        log.error('no graphs')
        sys.exit(-1)

    for n, gr in enumerate(clusters):
        if 0 == gr.number_of_nodes():
            log.error("no verticies")
            sys.exit(-1)
        dat = converter.clusters2blobs(gr)
        fname = paraview_file
        if '%' in paraview_file:
            fname = paraview_file%n
        write_data(dat, fname)
        log.info(fname)

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
            log.error("no verticies")
            return

        fname,ext=os.path.splitext(paraview_file)
        if '%' in fname:
            fname = fname%n

        alldat = converter.clusters2views(gr)
        for wpid, dat in alldat.items():
            pname = f'{fname}-plane{wpid}{ext}'
            write_data(dat, pname)
            log.info(pname)

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
        log.info(fname)
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
@click.option("--x0", default="0*cm",
              help="APA x location (with units)")
@click.option('-d', '--density', type=float, default=9.0,
              help="For samplings which care, specify target points per cc")
@click.argument("cluster-files", nargs=-1)
def bee_blobs(output, geom, rse, sampling, speed, t0, x0, density, cluster_files):
    '''
    Produce a Bee JSON file from a cluster file.
    '''
    if output is None:
        raise click.BadParameter("no output file provided")

    from . import tap, converter

    speed = unitify(speed)
    t0 = unitify(t0)
    x0 = unitify(x0)

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
        gr = converter.undrift_blobs(gr, speed, t0, x0)
        log.debug ("got %d" % gr.number_of_nodes())
        if 0 == gr.number_of_nodes():
            log.warning("skipping empty graph %s" % ctf)
            continue
        bnodes = nodes_oftype(gr,'b')
        bgr = gr.subgraph(bnodes)
        cluster_id = 0
        for bc in nx.connected_components(bgr):
            arr = converter.blobpoints(gr.subgraph(bc), sampling_func)
            if len(arr.shape) < 2:
                continue
            log.debug ("%s: %d points" % (ctf, arr.shape[0]))
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
    out_dir = os.path.dirname(output)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    json.dump(dat, open(output,'w', encoding="utf8"))


def divine_planes(nch):
    '''
    Return list of channels in each plane based on total.
    '''
    if nch == 2560:             # protodune
        return [400, 400, 400, 400, 480, 480]
    if nch == 8256:             # microboone
        return [2400, 2400, 3456]
    log.warning(f'not a canonical number of channels in a known detector: {nch}')
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
    Plot activity from a cluster file.
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
    log.debug(f'channel x slice array shape: {arr.shape}')
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
    im = ax.imshow(arr, cmap=cmap, interpolation='none', norm=LogNorm(), extent=extent, aspect='auto')
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
        log.error("colorbar complains, probably have zero data")
        log.error('total:', numpy.sum(arr))
        return

    ax.set_title(cluster_file)
    ax.set_xlabel("slice ID")
    ax.set_ylabel("channel IDs")
    fig.savefig(output)


@cli.command("blob-activity-stats")
@click.option('-o', '--output', default="/dev/stdout",
              help="Output to receive stats")
@click.option('-f', '--format', default="key=val",
              type=click.Choice(["key=val","json"]),
              help="Format for stats")
@click.option('--amin', default=0.0,
              help="Set the minimum activity to consider")
@click.argument("cluster-file")
def blob_activity_stats(output, format, amin, cluster_file):
    '''
    Return various statistics on blob and activity including:

    - atot :: total activity.

    - btot :: number of blobs.

    - qtot :: total of blob solved activity.

    - nbpix :: number of channel/slice pixels covered by blobs.

    - afound :: amount of activity captured by blobs.

    - pafound :: fraction of found activity.

    - amissed :: amount of activity missed by blobs.

    - pamissed :: fraction of missed activity.

    - pqtot :: ratio of total blob solved activity to total activity.

    - pqfound :: ratio of total blob solved activity to total activity covered by blobs.

    '''
    from . import tap, clusters, plots

    gr = list(tap.load(cluster_file))[0] # fixme
    cm = clusters.ClusterMap(gr)
    ahist = plots.activity(cm, amin)
    bhist = ahist.like()
    qhist = ahist.like()
    plots.blobs(cm, bhist)
    plots.blobs(cm, qhist, True)
    
    a = ahist.arr
    b = bhist.arr
    q = qhist.arr
    atot = float(numpy.sum(a))
    btot = float(numpy.sum(b))
    qtot = float(numpy.sum(q))

    nchan,nslice = a.shape
    # arr = numpy.ma.masked_where(sel(b), a)
    afound = float(numpy.sum(a[b >= 1.0]))
    amissed = float(numpy.sum(a[b < 1.0]))

    nbpix = float(numpy.sum(b >= 1))

    dat = dict(amin=amin, nchan=nchan, nslice=nslice,
               atot=atot, btot=btot, qtot=qtot,
               afound=afound, amissed=amissed,
               nbpix=nbpix,
               pamissed=amissed/atot, pafound=afound/atot,
               pqtot=qtot/atot, pqfound=qtot/afound)
    
    out = open(output, "w")
    if format == "json":
        out.write(json.dumps(dat, indent=4))
    if format == "key=val":
        for k,v in sorted(dat.items()):
            out.write(f'{k}={v}\n')

def parse_ranges(text):
    '''
    Parse list of integers and inclusive integer ranges.

    Text may look like "1,4-6,8,9" which returns [1,4,5,6,8,9].

    Ranges are inclusive.
    '''
    chans = list()
    for one in text.split(","):
        parts = list(map(int, one.split("-",1)))
        if len(parts) == 1:
            chans.append(parts[0])
        else:
            chans += range(parts[0], parts[1]+1)
    return chans
            
        

@cli.command("blob-activity-mask")
@click.option('-o', '--output',
              help="The output plot file name")
@click.option('-s', '--slices', nargs=2, type=int, 
              help="Narrow the range of slices")
@click.option('--slice-lines', default="",
              help="Draw lines at slices")
@click.option('--channel-lines', default="",
              help="Draw lines at channels")
@click.option('--found/--missed', default=True,
              help="Mask what blobs found or missed")
@click.option('--invert/--normal', default=False,
              help="Normally mask is black, zero is white or invert")
@click.option('--vmin', default=0.0,
              help="Set the minimum activity, below which is white")
@click.option('--amin', default=0.0,
              help="Set the minimum activity to consider")
@click.argument("cluster-file")
def blob_activity_mask(output, slices, channel_lines, slice_lines, found, invert, vmin, amin, cluster_file):
    '''Plot blobs as maskes on channel activity.

    By default, a mask is black and white is activity strictly less
    than --vmin and any other color is unmasked activity above or
    equal to vmin.  Use --invert to reverse the meaning of black and
    white.

    With --found the regions covered by blobs are masked (drawn as
    black by default).  A small --vmin=0.01 (eg) is recomended to
    distinquish regions of small activity from regions of zero
    activity.  Then, any remaining non-black/non-white color indicates
    activity that was not captured by any blob.  Using --invert may
    help to better see small regions with uncaptured activity.

    With --missed the mask is reversed and regions that are not
    covered by any blob are black.  Here a --vmin=0.01 (eg) will show
    show as white any region that has little to no activity and
    contributed to a blob.

    '''
    from . import tap, clusters, plots

    gr = list(tap.load(cluster_file))[0] # fixme
    cm = clusters.ClusterMap(gr)
    ahist = plots.activity(cm, amin)
    bhist = ahist.like()
    plots.blobs(cm, bhist)

    # make mask selector to be applied to bhist
    if found:
        sel = lambda p: p >= 1
        title="found mask"
    else:
        sel = lambda p: p < 1
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
    log.debug(f'extent: {extent}')

    fig,ax = plots.mask_blobs(a, b, sel, extent, vmin=vmin,
                              invert=invert, aspect='auto',
                              clabel="activity [ionization electrons / chan / slice]")

    # fixme: would be better to use rectangles

    if slice_lines:
        for sl in parse_ranges(slice_lines):
            ax.plot([sl, sl], [ahist.rangey[0], ahist.rangey[1]],
                    linewidth=1, color='gray', alpha=0.3)

    if channel_lines:
        for ch in parse_ranges(channel_lines):
            ax.plot([ahist.rangex[0], ahist.rangex[1]], [ch, ch],
                    linewidth=1, color='gray', alpha=0.1)

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
            log.debug(name, arr.shape)
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
                log.debug(f'rotate {code} {args}')
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
    
    
