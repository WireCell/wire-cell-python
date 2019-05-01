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


@cli.command("response-info")
@click.argument("json-file")
@click.pass_context
def response_info(ctx, json_file):
    '''
    Show some info about a field response file (.json or .json.bz2).
    '''
    import response.persist as per
    fr = per.load(json_file)
    print ("origin:%.2f cm, period:%.2f us, tstart:%.2f us, speed:%.2f mm/us, axis:(%.2f,%.2f,%.2f)" % \
           (fr.origin/units.cm, fr.period/units.us, fr.tstart/units.us, fr.speed/(units.mm/units.us), fr.axis[0],fr.axis[1],fr.axis[2]))
    for pr in fr.planes:
        print ("\tplane:%d, location:%.4fmm, pitch:%.4fmm" % \
               (pr.planeid, pr.location/units.mm, pr.pitch/units.mm))

@cli.command("convert-garfield")
@click.option("-o", "--origin", default="10.0*cm",
              help="Set drift origin (give units, eg '10*cm').")
@click.option("-s", "--speed", default="1.114*mm/us",
              help="Set nominal drift speed (give untis, eg '1.114*mm/us').")
@click.option("-n", "--normalization", default=0.0,
              help="Set normalization: 0:none, <0:electrons, >0:multiplicative scale.  def=0")
@click.option("-z", "--zero-wire-locs", default=[0.0,0.0,0.0], nargs=3, type=float,
              help="Set location of zero wires.  def: 0 0 0")
@click.argument("garfield-fileset")
@click.argument("wirecell-field-response-file")
@click.pass_context
def convert_garfield(ctx, origin, speed, normalization, zero_wire_locs,
                         garfield_fileset, wirecell_field_response_file):
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
    rflist = gar.load(garfield_fileset, normalization, zero_wire_locs)
    fr = res.rf1dtoschema(rflist, origin, speed)
    per.dump(wirecell_field_response_file, fr)




@cli.command("plot-garfield-exhaustive")
@click.option("-n", "--normalization", default=0.0,
              help="Set normalization: 0:none, <0:electrons, >0:multiplicative scale.  def=0")
@click.option("-z", "--zero-wire-locs", default=[0.0,0.0,0.0], nargs=3, type=float,
              help="Set location of zero wires.  def: 0 0 0")
@click.argument("garfield-fileset")
@click.argument("pdffile")
@click.pass_context
def plot_garfield_exhaustive(ctx, normalization, zero_wire_locs,
                                 garfield_fileset, pdffile):
    '''
    Plot all the Garfield current responses.
    '''
    import wirecell.sigproc.garfield as gar
    dat = gar.load(garfield_fileset, normalization, zero_wire_locs)
    import wirecell.sigproc.plots as plots
    plots.garfield_exhaustive(dat, pdffile)

@cli.command("plot-garfield-track-response")
@click.option("-g", "--gain", default=-14.0,
                  help="Set gain in mV/fC.")
@click.option("-s", "--shaping", default=2.0,
                  help="Set shaping time in us.")
@click.option("-t", "--tick", default=0.5,
                  help="Set tick time in us (0.1 is good for no shaping).")
@click.option("-p", "--tick-padding", default=0,
                  help="Number of ticks of zero ADC to pre-pad the plots.")
@click.option("-e", "--electrons", default=13300,
                help="Set normalization in units of electron charge.")
@click.option("-a", "--adc-gain", default=1.2,
                  help="Set ADC gain (unitless).")
@click.option("--adc-voltage", default=2.0,
                  help="Set ADC voltage range in Volt.")
@click.option("--adc-resolution", default=12,
                  help="Set ADC resolution in bits.")
@click.option("-n", "--normalization", default=-1,
                  help="Set normalization: 0:none, <0:electrons, >0:multiplicative scale.  def=-1")
@click.option("-z", "--zero-wire-locs", default=[0.0, 0.0, 0.0], nargs=3, type=float,
              help="Set location of zero wires.  def: 0 0 0")
@click.option("--ymin", default=-40.0,
                  help="Set Y min")
@click.option("--ymax", default=60.0,
                  help="Set Y max")
@click.option("--regions", default=0, type=int,
                  help="Set how many wire regions to use, default to all")
@click.option("--dump-data", default="", type=str,
                  help="Dump the plotted data in format given by extension (.json, .txt or .npz/.npy)")
@click.argument("garfield-fileset")
@click.argument("pdffile")
@click.pass_context
def plot_garfield_track_response(ctx, gain, shaping, tick, tick_padding, electrons,
                                     adc_gain, adc_voltage, adc_resolution,
                                     normalization, zero_wire_locs,
                                     ymin, ymax, regions,
                                     dump_data,
                                     garfield_fileset, pdffile):
    '''
    Plot Garfield response assuming a perpendicular track.

    Note, defaults are chosen to reproduce the "ADC Waveform with 2D
    MicroBooNE Wire Plane Model" plot for the MicroBooNE noise paper.
    '''
    import wirecell.sigproc.garfield as gar
    import wirecell.sigproc.response as res
    import wirecell.sigproc.plots as plots

    gain *= units.mV/units.fC
    shaping *= units.us
    tick *= units.us
    electrons *= units.eplus
    
    adc_gain *= 1.0                       # unitless
    adc_voltage *= units.volt
    adc_resolution = 1<<adc_resolution
    adc_per_voltage = adc_gain*adc_resolution/adc_voltage

    dat = gar.load(garfield_fileset, normalization, zero_wire_locs)

    if regions:
        print ("Limiting to %d regions" % regions)
        dat = [r for r in dat if abs(r.region) in range(regions)]

    uvw = res.line(dat, electrons)

    detector = ""
    if "ub_" in garfield_fileset:
        detector = "MicroBooNE"
    if "dune_" in garfield_fileset:
        detector = "DUNE"
    print ('Using detector hints: "%s"' % detector)

    nwires = len(set([abs(r.region) for r in dat])) - 1
    #msg = "%d electrons, +/- %d wires" % (electrons, nwires)
    msg=""

    fig,data = plots.plot_digitized_line(uvw, gain, shaping,
                                         adc_per_voltage = adc_per_voltage,
                                         detector = detector,
                                         ymin=ymin, ymax=ymax, msg=msg,
                                         tick_padding=tick_padding)
    print ("plotting to %s" % pdffile)
    fig.savefig(pdffile)

    if dump_data:
        print ("dumping data to %s" % dump_data)

        if dump_data.endswith(".npz"):
            import numpy
            numpy.savez(dump_data, data);
        if dump_data.endswith(".npy"):
            import numpy
            numpy.save(dump_data, data);
        if dump_data.endswith(".txt"):
            with open(dump_data,"wt") as fp:
                for line in data:
                    line = '\t'.join(map(str, line))
                    fp.write(line+'\n')
        if dump_data.endswith(".json"):
            import json
            open(dump_data,"wt").write(json.dumps(data.tolist(), indent=4))
                    



@cli.command("plot-response")
@click.argument("responsefile")
@click.argument("pdffile")
@click.pass_context
def plot_response(ctx, responsefile, pdffile):
    import response.persist as per
    import response.plots as plots

    fr = per.load(responsefile)
    plots.plot_planes(fr, pdffile)


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



@cli.command("channel-responses")
@click.option("-t","--tscale", default="0.5*us", type=str,
                  help="Scale of time axis in the histogram.")
@click.option("-s","--scale", default="1e-9*0.5/1.13312", type=str,
                  help="Scale applied to the samples.")
@click.option("-n","--name", default="pResponseVsCh",
                  help="Data name (eg, the TH2D name if using 'hist' schema")
@click.argument("infile")
@click.argument("outfile")
@click.pass_context
def channel_responses(ctx, tscale, scale, name, infile, outfile):
    '''Produce the per-channel calibrated response JSON file from a TH2D
    of the given name in the input ROOT file provided by the analysis.

    - tscale :: a number to multiply to the time axis of the histogram
      in order to bring the result into the WCT system of units.  It
      may be expressed as a string of an algebraic expression which
      includes symbols, eg "0.5*us".

    - scale :: a number multiplied to all samples in the histogram in
      order to make the sample value a unitless relative measure.  It
      may be expressed as a string of an algebraic expression which
      includes symbols, eg "0.5*us". For uBoone's initial
      20171006_responseWaveforms.root the appropriate scale is
      1e-9*0.5/1.13312 = 4.41267e-10
    '''
    import json
    import ROOT
    import numpy
    from root_numpy import hist2array

    tscale = eval(tscale, units.__dict__)
    scale = eval(scale, units.__dict__)

    tf = ROOT.TFile.Open(str(infile))
    assert(tf)
    h = tf.Get(str(name))
    if not h:
        click.echo('Failed to get histogram "%s" from %s' % (name, infile))
        sys.exit(1)

    arr,edges = hist2array(h, return_edges=True)

    arr *= scale
    tedges = edges[1]
    t0,t1 = tscale*(tedges[0:2])
    tick = t1-t0

    nchans, nticks = arr.shape
    channels = list()
    for ch in range(nchans):
        # reduce down to ~32 bit float precision to save file space
        res = [float("%.6g"%x) for x in arr[ch,:].tolist()] 
        one = [ch, res]
        channels.append(one)

    dat = dict(tick=tick, t0=t0, channels=channels)

    jtext = json.dumps(dat, indent=4)
    if outfile.endswith(".json.bz2"):
        import bz2
        bz2.BZ2File(outfile, 'w').write(jtext)
        return
    if outfile.endswith(".json.gz"):
        import gzip
        gzip.open(outfile, 'wb').write(jtext) # wb?
        return

    open(outfile, 'w').write(jtext)
    return


def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
