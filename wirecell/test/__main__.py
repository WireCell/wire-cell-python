import math
import json
import click
import functools
import dataclasses
from collections import defaultdict

import numpy
import matplotlib.pyplot as plt

from wirecell.util import ario, plottools
from wirecell.util.plottools import pages
from wirecell.util.cli import context, log
from wirecell.test import ssss
from wirecell.util.peaks import select_activity
from wirecell.util.codec import json_dumps

@context("test")
def cli(ctx):
    '''
    Wire Cell Test Commands
    '''
    pass

@cli.command("plot")
@click.option("-n", "--name", default="noise",
              help="The test name")
@click.argument("datafile")
@click.argument("output")
@click.pass_context
def plot(ctx, name, datafile, output):
    '''
    Make plots from file made by test_<test>.
    '''
    from importlib import import_module
    mod = import_module(f'wirecell.test.{name}')
    fp = ario.load(datafile)
    with plottools.pages(output) as out:
        mod.plot(fp, out)
    


def ssss_args(func):
    @click.option("--channel-ranges", default="0,800,1600,2560",
                  help="comma-separated list of channel idents defining ranges")
    @click.option("--nsigma", default=3.0,
                  help="Relative threshold on signal in units of number of sigma of noise width")
    @click.option("--nbins", default=50,
                  help="Number of bins over which to fit relative signal-splat difference")
    @click.option("-o",'--output', default='/dev/stdout')
    @click.argument("splat")
    @click.argument("signal")
    @functools.wraps(func)
    def wrapper(*args, **kwds):

        kwds["splat"] = ssss.load_frame(kwds.pop("splat"))
        kwds["signal"] = ssss.load_frame(kwds.pop("signal"))

        channel_ranges = kwds.pop("channel_ranges")
        if channel_ranges:
             channel_ranges = list(map(int,channel_ranges.split(",")))
             channel_ranges = [slice(*cr) for cr in zip(channel_ranges[:-1], channel_ranges[1:])]
        kwds["channel_ranges"] = channel_ranges
        return func(*args, **kwds)
    return wrapper


@cli.command("plot-ssss")
@click.option('--title', default='', help='extra title for plots')
@ssss_args
def plot_ssss(channel_ranges, nsigma, nbins, splat, signal, output, 
              title, **kwds):
    '''
    Perform the simple splat / sim+signal process comparison test and make plots.
    '''

    with pages(output) as out:

        ssss.plot_frames(splat, signal, channel_ranges, title)
        out.savefig()

        byplane = list()

        # Per channel range plots.
        for pln, ch in enumerate(channel_ranges):

            spl = select_activity(splat.frame, ch, nsigma)
            sig = select_activity(signal.frame, ch, nsigma)

            # Find the bbox that bounds the biggest splat object.
            biggest = spl.plats.sort_by("sums")[-1]
            bbox = spl.plats.bboxes[biggest]

            spl_act = spl.thresholded[bbox]
            sig_act = sig.thresholded[bbox]
            letter = "UVW"[pln]
            ssss.plot_plane(spl_act, sig_act, nsigma=nsigma,
                            title=f'{letter}-plane {title}')
            out.savefig()

            spl_qch = numpy.sum(spl.activity[bbox], axis=1)
            sig_qch = numpy.sum(sig.activity[bbox], axis=1)
            byplane.append((spl_qch, sig_qch))


        ssss.plot_metrics(byplane, nbins=nbins,
                          title=f'(splat - signal)/splat {title}')

        out.savefig()

@cli.command("ssss-metrics")
@ssss_args
@click.option("-p","--params",default=None,type=str,
              help="A depos.json parameter file associated with the splat and signal")
def ssss_metrics(channel_ranges, nsigma, nbins, splat, signal, output, params, **kwds):
    '''
    Write the simple splat / sim+signal process comparison metrics to file.

    The output JSON is an object with the key "metrics" holding the per-plane result.

    If a --params file is given it is merely attached at the "params" key.
    '''

    metrics = list()
    for pln, ch in enumerate(channel_ranges):
        spl = select_activity(splat.frame, ch, nsigma)
        sig = select_activity(signal.frame, ch, nsigma)

        biggest = spl.plats.sort_by("sums")[-1]
        bbox = spl.plats.bboxes[biggest]

        spl_qch = numpy.sum(spl.activity[bbox], axis=1)
        sig_qch = numpy.sum(sig.activity[bbox], axis=1)

        m = ssss.calc_metrics(spl_qch, sig_qch, nbins)
        metrics.append(dataclasses.asdict(m))

    if params:
        params = json.load(open(params))
    else:
        params = {}

    dat = dict(metrics=metrics, params=params)
    open(output,"w").write(json_dumps(dat, indent=4))


@cli.command("plot-metrics")
@click.option("-o","--output", default="metrics.pdf",
              help="PDF file in which to plot metrics")
@click.option("--coordinate-plane", default=None, type=int,
              help="Use given plane number as global coordinates plane, default uses per-plane coordinates")
@click.argument("files",nargs=-1)
def plot_metrics(output, coordinate_plane, files):
    '''Plot per-plane metrics from files.

    Files are as produced by ssss-metrics and must include a "params" key.
    '''

    # collect metrics by plane and order by angles.
    byplane = [defaultdict(list),defaultdict(list),defaultdict(list)]

    for fname in files:
        dat = json.load(open(fname))
        met = dat['metrics']
        par = dat['params']

        plane = par['plane_idx']
        detname = par['detector']

        if coordinate_plane is None: # consider wire-plane coordinates
            def add(k,v):
                byplane[plane][k].append(v)

            add('ty',  par['theta_y_wps'][plane])
            add('txz', par['theta_xz_wps'][plane])

            pmet = met[plane]
            add('ineff', pmet['ineff'])
            add('bias',  pmet['fit']['avg'])
            hi = pmet['fit']['hi']
            lo = pmet['fit']['lo']
            add('reso', 0.5*(hi+lo) )
            continue;

        # we want results from tracks defined for the plane with coordinate
        # system equated to the global system.  This is the MB SP-1 convention.
        if plane != coordinate_plane:
            continue

        for mplane, pmet in enumerate(met):
            def add(k,v):
                byplane[mplane][k].append(v)
            add('ty',  par['theta_y_wps'][plane])
            add('txz', par['theta_xz_wps'][plane])
            add('ineff', pmet['ineff'])
            add('bias',  pmet['fit']['avg'])
            hi = pmet['fit']['hi']
            lo = pmet['fit']['lo']
            add('reso', 0.5*(hi+lo) )
        
        
    parrs = list()
    for pdat in byplane:
        parrs.append({k:numpy.array(v) for k,v in pdat.items()})

    # gpick'ed from MB SP-1 paper
    pcolors = ('#58D453', '#7D99D1', '#D45853')

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    if coordinate_plane is None:
        fig.suptitle("Per-plane angles")
    else:
        letter = "UVW"[coordinate_plane]
        fig.suptitle(f'Global angles ({letter}-plane)')

    todeg = 180/numpy.pi
    # xlabs = [f'{txz}/{ty}' for txz,ty in zip(
    #     numpy.round(todeg*parrs[0]['txz']),
    #     numpy.round(todeg*parrs[0]['ty']))]
    xlabs = [f'{txz:.0f}' for txz in 
        numpy.round(todeg*parrs[0]['txz'])]

    axes[0].set_ylabel('Bias [%]')
    axes[1].set_ylabel('Resolution [%]')
    axes[2].set_ylabel('Inefficiency [%]')
    axes[2].set_xlabel('track angle [degree]')

    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
    x = numpy.arange(len(xlabs))
    width = 0.25
    def plotem(ax, key):
        for pind, parr in enumerate(parrs):
            xx = x + width*pind
            yy = 100*parr[key]
            letter = "UVW"[pind]
            rects = ax.bar(xx, yy, width, color=pcolors[pind], label=f'{letter} plane')
            #ax.bar_label(rects, padding=3) # to add numbers
            ax.axhline(y=0, linewidth=0.5, color='k')

    for mind, mkey in enumerate(['bias','reso','ineff']):
        plotem(axes[mind], mkey)
    axes[0].legend()

    plt.xticks(x, xlabs) #, rotation='vertical')
    # plt.xticks(rotation=30)
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(output)
            


def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
