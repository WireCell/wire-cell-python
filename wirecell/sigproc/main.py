#!/usr/bin/env python
'''
Export wirecell.sigproc functionality to a main Click program.
'''

import sys
import click

from wirecell import units

@click.group("sigproc")
@click.pass_context
def cli(ctx):
    '''
    Wire Cell Signal Processing Features
    '''


@cli.command("convert-garfield")
@click.option("-o", "--origin", default="10.0*cm",
              help="Set drift origin (give units, eg '10*cm').")
@click.option("-s", "--speed", default="1.114*mm/us",
              help="Set nominal drift speed (give untis, eg '1.114*mm/us').")
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

    origin = eval(origin, units.__dict__)
    speed = eval(speed, units.__dict__)
    rflist = gar.load(garfield_fileset)
    fr = res.rf1dtoschema(rflist, origin, speed)
    per.dump(wirecell_field_response_file, fr)




@cli.command("plot-garfield-exhaustive")
@click.argument("garfield-fileset")
@click.argument("pdffile")
@click.pass_context
def plot_garfield_exhaustive(ctx, garfield_fileset, pdffile):
    '''
    Plot all the Garfield current responses.
    '''
    import wirecell.sigproc.garfield as gar
    dat = gar.load(garfield_fileset)
    import wirecell.sigproc.plots as plots
    plots.garfield_exhaustive(dat, pdffile)

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
@click.option("--normalize/--no-normalize", default=True,
              help="Normalize responses so central collection wire gets 1 e-.")

@click.argument("garfield-fileset")
@click.argument("pdffile")
@click.pass_context
def plot_garfield_track_response(ctx, output, gain, shaping, tick, norm,
                                     adc_gain, adc_voltage, adc_resolution,
                                     normalize,
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

    dat = gar.load(garfield_fileset, normalize)
    uvw = res.line(dat, norm)

    detector = ""
    if "/ub_" in garfield_fileset:
        detector = "MicroBooNE"
    if "/dune_" in garfield_fileset:
        detector = "DUNE"

    fig,data = plots.plot_digitized_line(uvw, gain, shaping,
                                             adc_per_voltage = adc_per_voltage,
                                             detector = detector)
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


@cli.command("convert-noise-spectra")
@click.option("-f","--format", default="microboonev1",
                  help="Format of input file")
@click.argument("inputfile")
@click.argument("outputfile")
@click.pass_context
def convert_noise_spectra(ctx, format, inputfile, outputfile):
    '''
    Convert an file of noise spectra in some external format into WCT format.
    '''
    loader = None
    if format == "microboonev1":
        from wirecell.sigproc.noise.microboone import load_noise_spectra_v1
        loader = load_noise_spectra_v1
    #elif:...

    if not loader:
        click.echo('Unknown format: "%s"' % format)
        sys.exit(1)

    spectra = loader(inputfile)

    from wirecell.sigproc.noise import persist
    persist.dump(outputfile, spectra)

@cli.command("plot-noise-spectra")
@click.argument("spectrafile")
@click.argument("plotfile")
@click.pass_context
def plot_noise_spectra(ctx, spectrafile, plotfile):
    '''
    Plot contents of a WCT noise spectra file such as produced by
    the convert-noise-spectra subcommand.
    '''
    from wirecell.sigproc.noise import persist, plots
    spectra = persist.load(spectrafile)
    plots.plot_many(spectra, plotfile)


def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
