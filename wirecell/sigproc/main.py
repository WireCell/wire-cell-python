#!/usr/bin/env python
'''
Export wirecell.sigproc functionality to a main Click program.
'''

import click

from wirecell import units

@click.group("sigproc")
@click.pass_context
def cli(ctx):
    '''
    Wire Cell Signal Processing Features
    '''


@cli.command("convert-garfield")
@click.option("-o", "--origin", default=100.0,
              help="Set drift origin in units of mm.")
@click.option("-s", "--speed", default=1.114,
              help="Set nominal drift speed in units of mm/us.")
@click.argument("garfield-fileset")
@click.argument("wirecell-field-response-file")
@click.pass_context
def convert_garfield(ctx, origin, speed, garfield_fileset, wirecell_field_response_file):
    '''
    Convert an archive of a Garfield fileset (zip, tar, tgz) into a
    Wire Cell field response file (.json with optional .gz or .bz2
    compression).
    '''
    import garfield as gar
    import response as res
    import response.persist as per

    origin *= units.mm
    speed *= units.mm/units.us
    rflist = gar.load(garfield_fileset)
    fr = res.rf1dtoschema(rflist, origin, speed)
    per.dump(wirecell_field_response_file, fr)




@cli.command("plot-garfield-track-response")
@click.option("-o", "--output", default=None,
              help="Set output data file")
@click.option("-g", "--gain", default=14.0,
              help="Set gain in mV/fC.")
@click.option("-s", "--shaping", default=2.0,
              help="Set shaping time in us.")
@click.option("-t", "--tick", default=0.5,
              help="Set tick time in us (0.1 is good for no shaping).")
@click.option("-n", "--norm", default=16000,
              help="Set normalization in units of electron charge.")
@click.option("-a", "--adc-gain", default=1.2,
              help="Set ADC gain (unitless).")
@click.option("--adc-voltage", default=2.0,
              help="Set ADC voltage range in Volt.")
@click.option("--adc-resolution", default=12,
              help="Set ADC resolution in bits.")
@click.argument("garfield-fileset")
@click.argument("pdffile")
@click.pass_context
def plot_garfield_track_response(ctx, output, gain, shaping, tick, norm,
                                     adc_gain, adc_voltage, adc_resolution,
                                     garfield_fileset, pdffile):
    '''
    Plot Garfield response assuming a perpendicular track.
    '''
    import wirecell.sigproc.garfield as gar
    import wirecell.sigproc.response as res
    import wirecell.sigproc.plots as plots

    gain *= units.mV/units.fC
    shaping *= units.us
    tick *= units.us
    norm *= units.eplus
    
    adc_gain *= 1.0                       # unitless
    adc_voltage *= units.volt
    adc_resolution = 1<<adc_resolution
    adc_per_voltage = adc_gain*adc_resolution/adc_voltage

    dat = gar.load(garfield_fileset)
    uvw = res.line(dat, norm)

    fig,data = plots.plot_digitized_line(uvw, gain, shaping,
                                         adc_per_voltage = adc_per_voltage)
    fig.savefig(pdffile)


@cli.command("plot-electronics-response")
@click.option("-g", "--gain", default=14.0,
              help="Set gain in mV/fC.")
@click.option("-s", "--shaping", default=2.0,
              help="Set shaping time in us.")
@click.option("-t", "--tick", default=0.5,
              help="Set tick time in us (0.1 is good for no shaping).")
@click.argument("plotfile")
@click.pass_context
def plot_electronics_response(ctx, gain, shaping, tick, plotfile):
    '''
    Plot the electronics response function.
    '''
    gain *= units.mV/units.fC
    shaping *= units.us
    tick *= units.us
    import wirecell.sigproc.plots as plots
    fig = plots.one_electronics(gain, shaping, tick)
    fig.savefig(plotfile)


def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
