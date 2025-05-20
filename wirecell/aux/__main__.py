#!/usr/bin/env python3
'''
Main CLI to wirecell.aux.
'''

import json
import click
from wirecell.util import jsio
import tempfile
from pathlib import Path
import subprocess

from wirecell.util.cli import context, log
@context("aux")
def cli(ctx):
    '''
    wirecell-aux command line interface
    '''
    pass


@cli.command("run-idft")
@click.option("-e", "--epsilon", default=1e-6, type=float,
               help="Allowed error")
@click.option("-V", "--verbosity", default=0, type=int,
               help="Verbosity level")
@click.option("-o", "--output", default="run-idft-out.tar.bz2",
               help="Output PDF file")
@click.option("-p", "--plugin", default=None,
               help="WCT pluging holding the IDFT imp")
@click.option("-t", "--typename", default=None,
               help="Type or type:name of IDFT imp")
@click.option("-c", "--config", default=None, 
               help="Config file of IDFT operations (.json or .jsonnet)")
@click.option("-C", "--command", default="./build/aux/check_idft",
               help="The 'check_idft' command")
@click.argument("array_files", nargs=-1)
def run_idft(epsilon, verbosity, output, plugin, typename, config, command, array_files):
    """
    Perform DFT transforms with check_idft and numpy and compare.
    """
    from wirecell.aux import idft
    import numpy

    command = Path(command)
    if not command.exists():
        raise click.BadParameter(f'no such program: {command}')

    if not array_files:
        arrays = idft.gen_arrays()
        idft.save_arrays("run-idft-gen.tar.bz2", arrays)
        array_files = ["run-idft-gen.tar.bz2"]
    else:
        arrays = idft.get_arrays(array_files)

    if verbosity>1:
        log.info(" ".join(arrays.keys()))

    if not config:
        config = "run-idft-gen.json"
        idft.gen_config(config);

    py_out = dict()
    for one in jsio.load(config):
        if verbosity > 2:
            log.info(one)
        arr = arrays[one["src"]]
        op = getattr(idft, one["op"], None)
        if op is None:          # literally no-op :D
            py_out[one["dst"]] = arr
        else:
            py_out[one["dst"]] = op(arr)

    cmd = [command, "-o", output, "-c", config]
    if plugin:
        cmd += ["-p", plugin]
    if typename:
        cmd += ["-t", typename]
    cmd += array_files

    cmdstr = " ".join(cmd)
    if verbosity>0:
        log.info(f"\nRunning: {cmdstr}...\n")
    try:
        subprocess.check_output(cmd)
    except Exception:
        cmdstr = ' '.join(cmd)
        log.error(f'Failed to run: "{cmdstr}"')
        log.error('''Perhaps try:
        - give "--command /path/to/check_idft" option
        - run "./wcb --tests --target=check_idft to install
        - examine above error messages''')
        raise
    if verbosity>0:
        log.info(f"\n...done\n")
    wc_out = idft.get_arrays([output])

    keys = list(set(list(wc_out.keys()) + list(py_out.keys())))
    keys.sort()
    err = 0
    for key in keys:
        p = py_out.get(key, None);
        w = wc_out.get(key, None);

        def summary():
            if verbosity < 1:
                return
            log.info (f'\tconfig: {one}')
            log.info (f'\tshapes: numpy:{p.shape} wirecell:{w.shape}')
            log.info (f'\tdtypes: numpy:{p.dtype} wirecell:{w.dtype}')
            if verbosity < 2:
                return
            log.info (f'\tsum: numpy:{numpy.sum(p)}, wirecell:{numpy.sum(w)}')
            log.info (f'\tnumpy:\n{p}\n\twirecell:\n{w}')
            

        def fail(what):
            nonlocal err
            err += 1
            log.error(f'fail: {key}: {what} (error #{err})')
            summary()

        if p is None:
            fail("missing number array")
            continue
        if w is None:
            fail("missing wirecell array")
            continue
        if p.shape != w.shape:
            fail('shape mismatch')
            continue
        if p.dtype != w.dtype:
            fail('dtype mismatch')
            continue
        adiff = numpy.abs(p-w)
        asum = numpy.abs(p+w)
        indmax = numpy.unravel_index(numpy.argmax(adiff), adiff.shape)
        avg = 0.5*(p[indmax] + w[indmax])
        amax = adiff[indmax]

        if avg != 0:
            amax /= avg

        if amax > epsilon:
            fail(f'large max diff: {amax}')
            continue

        l1 = numpy.sum(adiff) / numpy.sum(asum)
        if l1 > epsilon:
            fail(f'large L1: {l1}')
            continue

        log.info(f'pass: {key} {p.shape} {p.dtype}')
        summary()

    if err == 1:
        log.info (f'1 error')
    else:
        log.info (f'{err} errors')


    # run check_idft, make output file
    # read output file
    # compare wc and py array by array


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
    from wirecell.aux import sysinfo

    # data = sysinfo.asdict()
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
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

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
    
