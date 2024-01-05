import math
import click
import numpy
import functools
import dataclasses

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
def ssss_metrics(channel_ranges, nsigma, nbins, splat, signal, output, **kwds):
    '''
    Write the simple splat / sim+signal process comparison metrics to file.
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

    open(output,"w").write(json_dumps(metrics, indent=4))


def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
