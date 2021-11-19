#!/usr/bin/env python3
'''
Main CLI to wirecell.aux.
'''

import json
import click
from wirecell.util import ario, plottools
import numpy
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
import subprocess
from wirecell.aux import idft, sysinfo
from matplotlib.backends.backend_pdf import PdfPages

@click.group()
@click.pass_context
def cli(ctx):
    '''
    wirecell-aux command line interface
    '''
    ctx.ensure_object(dict)


@cli.command("run-idft-bench")
@click.option("-o", "--output", default="idft-bench.json",
               help="Output PDF file")
@click.option("-p", "--plugin", default=None,
               help="WCT pluging holding the IDFT imp")
@click.option("-t", "--typename", default=None,
               help="Type or type:name of IDFT imp")
@click.option("-c", "--config", default=None,
               help="Config file of IDFT imp")
@click.argument("program")
def run_idft_bench(output, plugin, typename, config, program):
    '''
    Run the check_idft_bench, augmenting the results with host info
    '''

    data = sysinfo.asdict()
    opath = Path(output);
    with tempfile.TemporaryDirectory(prefix=opath.stem) as tdir:
        jfile = tdir + "/idft-bench.json" 
        args = [program, "-o", jfile]
        if config:
            args += ["-c", config]
        if typename:
            args += ["-t", typename]
        if plugin:
            args += ["-p", plugin]
        subprocess.check_output(args)
        dat = json.loads(open(jfile).read());
        dat[0]["sysinfo"] = sysinfo.asdict();
        open(output, "w").write(json.dumps(dat, indent=4));


@cli.command("plot-idft-bench")
@click.option("-o", "--output", default="idft-bench.pdf",
               help="Output PDF file")
@click.argument("inputs", nargs=-1)
def plot_idft_bench(output, inputs):
    '''
    Make plots from one or more check_idft_bench output JSON files.
    '''
    dats = [idft.load(inp) for inp in inputs]
    with PdfPages(output) as pdf:
        for func_name in ["fwd1d", "fwd2d", "fwd1b0", "fwd1b1"]:
            for measure in ["time", "clock"]:
                idft.plot_time(dats, func_name, measure)
                pdf.savefig()
                plt.close()
                
                idft.plot_plan_time(dats, func_name, measure)
                pdf.savefig()
                plt.close()
    
def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
