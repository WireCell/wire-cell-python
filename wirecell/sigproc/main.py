#!/usr/bin/env python
'''
Export wirecell.sigproc functionality to a main Click program.
'''

import click

@click.group("sigproc")
@click.pass_context
def cli(ctx):
    '''
    Wire Cell Signal Processing Features
    '''


@cli.command("convert-garfield")
@click.argument("garfield-fileset")
@click.argument("wirecell-field-response-file")
@click.pass_context
def convert_garfield(ctx, garfield_fileset, wirecell_field_response_file):
    '''
    Convert an archive of a Garfield fileset (zip, tar, tgz) into a
    Wire Cell field response file (.json with optional .gz or .bz2
    compression).
    '''
    import wirecell.sigproc.garfield as gar
    import wirecell.sigproc.response as res
    import wirecell.sigproc.response.persist as per

    rflist = gar.load(garfield_fileset)
    fr = res.rf1dtoschema(rflist)
    per.dump(wirecell_field_response_file, fr)


@cli.command("plot-field-response")
@click.argument("wcfrfile")
@click.argument("pdffile")
@click.pass_context
def plot_field_response(ctx, wcfrfile, pdffile):
    import wirecell.sigproc.response.persist as per
    import wirecell.sigproc.response.plots as plt

    fr = per.load(wcfrfile)
    # ...


@cli.command("plot-track-response")
@click.option("-o", "--output", default=None,
              help="Set output data file")
@click.option("-g", "--gain", default=14.0,
              help="Set gain.")
@click.option("-s", "--shaping", default=2.0,
              help="Set shaping time in us.")
@click.option("-t", "--tick", default=0.5,
              help="Set tick time in us (0.1 is good for no shaping).")
@click.option("-n", "--norm", default=16000,
              help="Set normalization in units of electron charge.")
@click.argument("garfield-fileset")
@click.argument("pdffile")
@click.pass_context
def plot_track_response(ctx, output, gain, shaping, tick, norm,
                            garfield_fileset, pdffile):
    import wirecell.sigproc.garfield as gar
    import wirecell.sigproc.response as res
    import wirecell.sigproc.plots as plots
    from wirecell.sigproc import units

    shaping *= units.us
    tick *= units.us
    norm *= units.electron_charge

    dat = gar.load(garfield_fileset)

    uvw = res.line(dat, norm)

    adc_gain = 1.2          # post amplifier gain, was 1.1 for a while
    adc_bin_range = 4096.0
    adc_volt_range = 2000.0
    adc_per_mv = adc_gain*adc_bin_range/adc_volt_range

    fig,data = plots.plot_digitized_line(uvw, gain, shaping,
                                         adc_per_mv = adc_per_mv)
    fig.savefig(pdffile)


def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
