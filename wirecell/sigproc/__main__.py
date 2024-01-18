#!/usr/bin/env python
'''
Export wirecell.sigproc functionality to a main Click program.
'''
import re
import sys
import click
import numpy
from wirecell import units

from wirecell.util.cli import context, log

@context("sigproc")
def cli(ctx):
    '''
    Wire Cell Signal Processing Features
    '''
    pass

@cli.command("fr2npz")
@click.option("-g", "--gain", default=0.0, type=float,
                  help="Set gain in mV/fC.")
@click.option("-s", "--shaping", default=0.0, type=float,
                  help="Set shaping time in us.")
@click.argument("json-file")
@click.argument("npz-file")
def fr2npz(gain, shaping, json_file, npz_file):
    '''
    Convert field response file to numpy (.json or .json.bz2 to .npz)

    If gain and shaping are non zero then convolve each field response
    function with the corresponding electronics response function.

    Result holds a number of arrays and scalar values.

    Arrays:

        - resp[012] :: one 2D array for each plane.  A wire region is
          10 pixels wide.  Each row of pixels represents the average
          field response between the two original drift paths bounding
          the row.  The 2D array also makes explicit the
          flipped-symmetry that the original field response file
          leaves implicit.  The columns mark time.  Note, to make a
          per-wire sub-array of all impact rows #3 (counting from 0)
          you can use Numpy indexing like: dat['resp2'][3::10,:]

        - bincenters[012] :: the pitch location in mm of the row
          centers

        - pitches :: the nominal wire pitch for each plane as used in
          the Garfield simulation.

        - locations :: the locations along the drift direction of each
          of the planes.

        - eresp :: electronics response function, if gain/shaping given

        - espec :: the FFT of this, if gain/shaping given

    Scalar values:

        - origin :: in mm of where the drifts start in the same axis
          as the locations

        - tstart :: time when drift starts

        - period :: the sampling period in ns of the field response
          (ie, width of resp columns)

        - speed :: in mm/ns of the nominal electron drift speed used
          in the Garfield calculation.

        - gain : the passed in gain

        - shaping :: the passed in shaping time

    '''
    import wirecell.sigproc.response.persist as per
    import wirecell.sigproc.response.arrays as arrs

    fr = per.load(json_file)
    # when json_file really names a detector, that detector may have more than
    # one field response file - as in uboone.  Here, we only support the first.
    if isinstance(fr, list):
        fr = fr[0]              
    gain = units.mV/units.fC
    shaping *= units.us
    dat = arrs.fr2arrays(fr, gain, shaping)
    numpy.savez(npz_file, **dat)


@cli.command("frzero")
@click.option("-n", "--number", default=0,
              help="Number of strip to keep, default keeps just zero strip")              
@click.option("-o", "--output",
              default="/dev/stdout",
              help="Output WCT file (.json or .json.bz2, def: stdout)")
@click.argument("infile")
@click.pass_context
def frzero(ctx, number, output, infile):
    '''
    Given a WCT FR file, make a new one with off-center wires zeroed.
    '''
    import wirecell.sigproc.response.persist as per
    import wirecell.sigproc.response.arrays as arrs
    fr = per.load(infile)
    for pr in fr.planes:
        for path in pr.paths:

            wire = int(path.pitchpos / pr.pitch)
            if abs(wire) <= number:
                log.info(f'keep wire: {wire}, pitch = {path.pitchpos} / {pr.pitch}')
                continue

            nc = len(path.current)
            for ind in range(nc):
                path.current[ind] = 0;
    per.dump(output, fr)

@cli.command("response-info")
@click.argument("json-file")
@click.pass_context
def response_info(ctx, json_file):
    '''
    Show some info about a field response file (.json or .json.bz2).
    '''
    import wirecell.sigproc.response.persist as per
    fr = per.load(json_file)
    log.info ("origin:%.2f cm, period:%.2f us, tstart:%.2f us, speed:%.2f mm/us, axis:(%.2f,%.2f,%.2f)" % \
           (fr.origin/units.cm, fr.period/units.us, fr.tstart/units.us, fr.speed/(units.mm/units.us), fr.axis[0],fr.axis[1],fr.axis[2]))
    for pr in fr.planes:
        log.info ("\tplane:%d, location:%.4fmm, pitch:%.4fmm" % \
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
@click.option("-d", "--delay", default=0, type=int,
              help="Set additional delay of bins in the output field response.  def=0")
@click.argument("garfield-fileset")
@click.argument("wirecell-field-response-file")
@click.pass_context
def convert_garfield(ctx, origin, speed, normalization, zero_wire_locs,
                    delay, garfield_fileset, wirecell_field_response_file):
    '''
    Convert an archive of a Garfield fileset (zip, tar, tgz) into a
    Wire Cell field response file (.json with optional .gz or .bz2
    compression).
    '''
    import wirecell.sigproc.garfield as gar
    from wirecell.sigproc.response import rf1dtoschema
    import wirecell.sigproc.response.persist as per

    origin = eval(origin, units.__dict__)
    speed = eval(speed, units.__dict__)
    rflist = gar.load(garfield_fileset, normalization, zero_wire_locs, delay)
    fr = rf1dtoschema(rflist, origin, speed)
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
@click.option("--elec-type", default="cold", type=str,
                  help="Set electronics type [cold | warm] (def: cold).")
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
                                     elec_type, adc_gain, adc_voltage, adc_resolution,
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
        log.debug ("Limiting to %d regions" % regions)
        dat = [r for r in dat if abs(r.region) in range(regions)]

    uvw = res.line(dat, electrons)

    detector = ""
    if "ub_" in garfield_fileset:
        detector = "MicroBooNE"
    if "dune_" in garfield_fileset:
        detector = "DUNE"
    log.debug ('Using detector hints: "%s"' % detector)

    nwires = len(set([abs(r.region) for r in dat])) - 1
    #msg = "%d electrons, +/- %d wires" % (electrons, nwires)
    msg=""

    fig,data = plots.plot_digitized_line(uvw, gain, shaping,
                                         tick = tick,
                                         elec_type = elec_type,
                                         adc_per_voltage = adc_per_voltage,
                                         detector = detector,
                                         ymin=ymin, ymax=ymax, msg=msg,
                                         tick_padding=tick_padding)
    log.debug ("plotting to %s" % pdffile)
    fig.savefig(pdffile)

    if dump_data:
        log.debug ("dumping data to %s" % dump_data)

        if dump_data.endswith(".npz"):
            numpy.savez(dump_data, data);
        if dump_data.endswith(".npy"):
            numpy.save(dump_data, data);
        if dump_data.endswith(".txt"):
            with open(dump_data,"wt") as fp:
                for line in data:
                    line = '\t'.join(map(str, line))
                    fp.write(line+'\n')
        if dump_data.endswith(".json"):
            import json
            open(dump_data,"wt").write(json.dumps(data.tolist(), indent=4))


@cli.command("plot-response-compare-waveforms")
@click.option("-p", "--plane", default=0,
              help="Plane")
@click.option("--irange", default='0',
              help="Impact range as comma separated integers")
@click.option("--trange", default='0,70',
              help="Set time range in us as comma pair. def: 0,70")
@click.argument("responsefile1")
@click.argument("responsefile2")
@click.argument("outfile")
@click.pass_context
def plot_response_compare_waveforms(ctx, plane, irange, trange, responsefile1, responsefile2, outfile):
    '''
    Plot common response waveforms from two sets
    '''
    import wirecell.sigproc.response.persist as per
    import wirecell.sigproc.response.plots as plots

    irange = list(map(int, irange.split(',')))
    trange = list(map(int, trange.split(',')))

    colors = ["red","blue"]
    styles = ["solid","solid"]

    import matplotlib.pyplot as plt

    def plot_paths(rfile, n):
        fr = per.load(rfile)
        pr = fr.planes[plane]
        log.debug(f'{colors[n]} {rfile}: plane={plane} {len(pr.paths)} paths:')
        for ind in irange:
            path = pr.paths[ind]
            tot_q = numpy.sum(path.current)*fr.period
            dt_us = fr.period/units.us
            tot_es = tot_q / units.eplus
            log.debug (f'\t{ind}: {path.pitchpos:f}: {len(path.current)} samples, dt={dt_us:.3f} us, tot={tot_es:.3f} electrons')
            plt.gca().set_xlim(*trange)

            times = plots.time_linspace(fr, plane)

            plt.plot(times/units.us, path.current, color=colors[n], linestyle=styles[n])
        
    plot_paths(responsefile1,0)
    plot_paths(responsefile2,1)
    plt.savefig(outfile)




@cli.command("plot-response")
@click.option("--region", default=None, type=float,
              help="Set a region to demark as 'electrode 0'. def: none")
@click.option("--trange", default='0,70',
              help="Set time range in us as comma pair. def: 0,70")
@click.option("--title", default='Response:',
              help="An initial distinguishing title")
@click.option("--reflect/--no-reflect", default=True,
              help="Apply symmetry reflection")
@click.argument("responsefile")
@click.argument("outfile")
@click.pass_context
def plot_response(ctx, responsefile, outfile, region, trange, title, reflect):
    '''
    Plot per plane responses.
    '''
    import wirecell.sigproc.response.persist as per
    import wirecell.sigproc.response.plots as plots

    trange = list(map(int, trange.split(',')))
    fr = per.load(responsefile)
    plots.plot_planes(fr, outfile, trange, region, reflect, title)


@cli.command("plot-response-conductors")
@click.option("--trange", default='0,70',
              help="Set time range in us as comma pair. def: 0,70")
@click.option("--title", default='Response:',
              help="An initial distinguishing title")
@click.option("--log10/--no-log10", default=False,
              help="Use 'signed log10' else linear scale")
@click.option("--regions", default=None,
              help="Comma separated list of regions, default is all")
@click.argument("responsefile")
@click.argument("outfile")
@click.pass_context
def plot_response(ctx, trange, title, log10, regions, responsefile, outfile):
    '''
    Plot per-conductor (wire/strip) respnonses.
    '''
    import wirecell.sigproc.response.persist as per
    import wirecell.sigproc.response.plots as plots

    trange = list(map(int, trange.split(',')))
    if regions:
        regions = list(map(int, regions.split(',')))
    fr = per.load(responsefile)
    plots.plot_conductors(fr, outfile, trange, title, log10, regions)



@cli.command("plot-spectra")
@click.argument("responsefile")
@click.argument("outfile")
@click.pass_context
def plot_spectra(ctx, responsefile, outfile):
    '''
    Plot per plane response spectra.
    '''
    import wirecell.sigproc.response.persist as per
    import wirecell.sigproc.response.plots as plots

    fr = per.load(responsefile)
    plots.plot_specs(fr, outfile)


@cli.command("plot-electronics-response")
@click.option("-g", "--gain", default=14.0,
              help="Set gain in mV/fC.")
@click.option("-s", "--shaping", default=2.0,
              help="Set shaping time in us.")
@click.option("-t", "--tick", default=0.5,
              help="Set tick time in us (0.1 is good for no shaping).")
@click.option("-e", "--electype", default="cold",
              help="Set electronics type [cold | warm] (def: cold).")
@click.argument("plotfile")
@click.pass_context
def plot_electronics_response(ctx, gain, shaping, tick, electype, plotfile):
    '''
    Plot the electronics response function.
    '''
    gain *= units.mV/units.fC
    shaping *= units.us
    tick *= units.us
    import wirecell.sigproc.plots as plots
    fig = plots.one_electronics(gain, shaping, tick, electype)
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
    elif format == "icarusv1-incoherent":
        from wirecell.sigproc.noise.icarus import load_noise_spectra
        loader = load_noise_spectra
    elif format == "icarusv1-coherent":
        from wirecell.sigproc.noise.icarus import load_coherent_noise_spectra
        loader = load_coherent_noise_spectra

    if not loader:
        click.echo('Unknown format: "%s"' % format)
        sys.exit(1)

    spectra = loader(inputfile)

    from wirecell.sigproc.noise import persist
    persist.dump(outputfile, spectra)

@cli.command("convert-electronics-response")
@click.argument("inputfile")
@click.argument("outputfile")
@click.pass_context
def convert_noise_spectra(ctx, inputfile, outputfile):
    '''
    Convert a table of electronics response function in some external format into WCT format.
    '''
    from wirecell.sigproc.response import load_text_electronics_response
    loader = load_text_electronics_response
    if not loader:
        click.echo('Unknown format: "%s"' % format)
        sys.exit(1)

    elecresp = loader(inputfile)

    from wirecell.sigproc.response import persist
    persist.dump(outputfile, elecresp)

from wirecell.util.cli import jsonnet_loader
from wirecell.util import jsio
@cli.command("plot-noise-spectra")
@click.option("-z", "--zero-suppress", is_flag=True, default=False, help="Set zero frequency bin to zero")
@jsonnet_loader("spectra")
@click.argument("plotfile")
@click.pass_context
def plot_noise_spectra(ctx, zero_suppress, spectra, plotfile, **kwds):
    '''
    Plot contents of a WCT noise spectra file such as produced by
    the convert-noise-spectra subcommand or a NoiseModeler.
    '''
    from wirecell.sigproc.noise import plots
    plots.plot_many(spectra, plotfile, zero_suppress)



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


@cli.command("plot-configured-spectra")
@click.option("-t","--type", default="GroupNoiseModel", help="component type name")
@click.option("-n","--name", default="inco", help="component instance name")
@click.option("-d","--data", default="spectra", help="the key in .data holding configured spectra")
@click.option("-c", "--coupling", type=click.Choice(["ac","dc"]), default="dc",
              help="To suppress zero-frequency bin ('ac') or not ('dc'), def=dc")
@click.option("-o","--output", default="/dev/stdout", help="output for plots")
@click.argument("cfgfile")
@click.argument("output")
def configured_spectra(type, name, data, coupling, output, cfgfile):
    '''
    Plot configured spectra
    '''
    from wirecell.util import jsio
    from wirecell.sigproc.noise.plots import plot_many

    got = None
    for one in jsio.load(cfgfile):
        if one['type'] == type and one['name'] == name:
            got=one['data'][data]
            break
    if got is None:
        raise click.BadParameter(f'failed to find node {type}:{name}')

    zero_suppress = coupling == "ac"
    plot_many(got, output, zero_suppress)
    

@cli.command("fwd")
@click.option("--plots", type=str, default=None,
              help="comma-separated list regex to match on plot categories")
@click.option("-o","--output", default="/dev/stdout", help="output for plots")
@click.argument("detector")
def fwd(plots, output, detector):
    '''
    Simple exercise of ForWaRD
    '''

    if plots:
        plots = [re.compile(p) for p in plots.split(",")]
    def will_plot(key):
        if not plots:
            return True         # all
        for p in plots:
            if p.match(key):
                return True
        return False

    from . import fwd
    from wirecell.util.plottools import pages
    import matplotlib.pyplot as plt

    # ADC sample time basis
    T=0.5*units.us
    N=1024
    Fmax = 1.0/T
    times = fwd.make_times(N, T)

    # FR and thus DR time basis is 5x more fine than ADC
    dr_fine = fwd.DetectorResponse("det fine", detector)
    print(dr_fine)
    nrebin = int(T/dr_fine.T)

    # A rebinned dr but that does not convolve in the slower sampling.
    dr = fwd.rebin(dr_fine, nrebin, "det tick")
    dr.er = fwd.rebin(dr_fine.er, nrebin, "elec tick")
    dr.fr = fwd.rebin(dr_fine.fr, nrebin, "field tick")
    print (dr)

    # The noise "signal"
    noi = fwd.Noise("noise", times, detector)

    # Display units
    dunits = fwd.unit_info()

    # Simple diffusion model
    smears = list()
    sigmas = numpy.array([1,2,4,8])*units.us
    for sigma in sigmas:
        # smear = fwd.gauss_wave(times, sigma*units.us);
        smear = fwd.gauss_wave(times, sigma, times[N//2]);
        smears.append(fwd.Signal(f'{sigma/units.us} us smear', times, smear))

    # Simple track signal model: MIP tracks of different sample time duration
    speed = 1.6 * units.mm / units.us
    mip_ione = 55000/units.cm * speed
    qstep = mip_ione * T
    track_widths = numpy.array([2, 10, 100, 400])*units.us
    squares = [fwd.square(times, w, name=f'{w/units.us} us track') * qstep for w in track_widths]

    # Simple multi-Gaussian "blips" signal model
    blips = list()
    blip_counts = (2, 10, 100)
    for n in blip_counts:
        wave = fwd.randgauss_wave(times, n=n,
                                  amprange=(0.5*qstep, 2*qstep),
                                  sigrange=(0.1*units.us, 0.5*units.us),
                                  timerange=(200*units.us, 300*units.us))
        sig = fwd.Signal(f'{n} blips', times, numpy.sum(wave, axis=0))
        blips.append(sig)

    # Note, order here matches forward.org presentation.
    # Changing it will break rebuilding that file.
    with pages(output) as out:

        # Responses
        if will_plot("responses"):
            for one in [dr_fine, dr]:
                fwd.plot_convo(one.fr, one.er, one,
                               dunits['field'], dunits['elec'], dunits['det'],
                               flimiter = fwd.range_limiter(0, Fmax / 2))
                out.savefig()
                plt.clf()

        # Noise
        if will_plot("noise"):
            fwd.plot_noise(noi, dunits['noise'])
            out.savefig()
            plt.clf()

        # Several pages of diffusion of tracks 
        square_sigs = list()    # collect
        for square in squares:
            sigs = [fwd.Convolution(square.name, square, r, "same") for r in smears]
            square_sigs.append(sigs)

            if will_plot("square_sigs"):
                fwd.plot_convo(square, smears, sigs,
                               dunits['ionized'], dunits['diffusion'], dunits['drifted'],
                               flimiter = fwd.range_limiter(0, Fmax / 2))
                out.savefig()
                plt.clf()

        # Track (x) response
        square_meas_quiet = list() # collect
        for sigs, tw in zip(square_sigs, track_widths):
            meas = [fwd.Convolution(f'{tw/units.us} us track', sig, dr) for sig in sigs]
            square_meas_quiet.append(meas)
            if will_plot("square_measures_quiet"):
                fwd.plot_convo(sigs, dr, meas,
                               dunits['drifted'], dunits['det'], dunits['quiet'],
                               flimiter = fwd.range_limiter(0, Fmax / 2))
                out.savefig()
                plt.clf()

        for sigs in square_meas_quiet:
            if will_plot("square_measures_noise"):
                sig = sigs[1]
                tot = sig + noi
                tot.name = sig.name + " + noise"
                fwd.plot_signal_noise(sig, noi, tot,
                                      dunits['noise'],
                                      flimiter = fwd.range_limiter(0, Fmax / 2))
                out.savefig()
                plt.clf()


        # Several pages of diffusion of different number of Gaussian "blips"
        blips_sigs = list()
        for sig in blips:
            sigs = [fwd.Convolution(sig.name, sig, r, "same") for r in smears]        
            blips_sigs.append(sigs)
            if will_plot("blips_sigs"):
                fwd.plot_convo(sig, smears, sigs,
                               dunits['ionized'], dunits['diffusion'], dunits['drifted'],
                               flimiter = fwd.range_limiter(0, Fmax / 2))
                out.savefig()
                plt.clf()

        blips_meas_quiet = list() # collect
        for sigs, nblips in zip(blips_sigs, blip_counts):
            meas = [fwd.Convolution(f'{nblips} blips', sig, dr) for sig in sigs]
            blips_meas_quiet.append(meas)
            if will_plot("blips_measures_quiet"):
                fwd.plot_convo(sigs, dr, meas,
                               dunits['drifted'], dunits['det'], dunits['quiet'],
                               flimiter = fwd.range_limiter(0, Fmax / 2))
                out.savefig()
                plt.clf()
                

        for sigs in blips_meas_quiet:
            if will_plot("blips_measures_noise"):
                sig = sigs[-1]
                tot = sig + noi
                tot.name = sig.name + " + noise"
                fwd.plot_signal_noise(sig, noi, tot,
                                      dunits['noise'],
                                      flimiter = fwd.range_limiter(0, Fmax / 2))
                out.savefig()
                plt.clf()
                
        
# next
# - [x] factor gauss smear to function and out of plot_signal_demo
# - [x] factor out demo and plotting from plot_signal_demo
# - [x] convolve signal with response and plot wave/spec
# - [x] understand noise mode/mean/energy
# - [x] understand noise normalization
# - [x] implement noise sim
# - [x] add simple signal sim (track and blip)
# - [x] plot signal spectrum as function of track length 
# - [ ] understand signal normalization
# - [ ] signal spec + noise spec + FoRD
# - [ ] plot (normalized) signal/noise spectra with shrinkage coefficients overlayed
                

def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
