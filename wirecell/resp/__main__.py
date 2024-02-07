#!/usr/bin/env python3
'''
Main CLI to wirecell.resp.
'''

import click
from wirecell.util.fileio import load as source_loader
from wirecell import units
from wirecell.util.functions import unitify
import numpy
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

cmddef = dict(context_settings = dict(help_option_names=['-h', '--help']))

@click.group(**cmddef)
@click.pass_context
def cli(ctx):
    '''
    wirecell-resp command line interface
    '''
    ctx.ensure_object(dict)


@cli.command("gf2npz")
@click.option("-o", "--output",
              type=click.Path(dir_okay=False, writable=True),
              help="Name the output NPZ file")
@click.option("--origin", type=str,
              help="Set drift origin (give units, eg '10*cm').")
@click.option("--speed", type=str,
              help="Set drift speed at start of response (give untis, eg '1.114*mm/us').")
@click.argument("dataset")
def gf2npz(output, origin, speed, dataset):
    '''
    Convert a Garfield data set to a "WCT response NPZ" file.
    '''
    if not all([speed, origin]):
        raise ValueError("You MUST give --speed and --origin")

    from wirecell.resp.garfield import (
        dataset_asdict, dsdict2arrays)
    source = source_loader(dataset, pattern="*.dat")
    ds = dataset_asdict(source)

    origin = eval(origin, units.__dict__)
    speed = eval(speed, units.__dict__)
    arrs = dsdict2arrays(ds, speed, origin)
    numpy.savez(output, **arrs)

@cli.command("gf-info")
@click.argument("dataset")
def gf_info(dataset):
    '''
    Give info about a garfield dataset
    '''
    from wirecell.resp.garfield import (
        dataset_asdict, dsdict_dump)
    source = source_loader(dataset, pattern="*.dat")
    ds = dataset_asdict(source)
    dsdict_dump(ds)


@cli.command("condition")
@click.option("-P", "--period", default=None, type=str,
              help="Override the sampling period given in the frfile, eg '100*ns'")
# @click.option("-e", "--error", default=1e-6,
#               help="Precision by which integer and rationality conditions are judged")
@click.option("-r", "--rolloff", default=None, type=float,
              help="Add a roll-off to the spectrum starting at given frequency in units of Nyquist frequency.")
@click.option("-o", "--output", default="/dev/stdout",
              help="File in which to write the result")
@click.argument("frfile")
def condition(period, rolloff, output, frfile):
    '''
    Condition an FR for resampling.

    This will write a new FR file the various transformations.

    -P|--period will force set a new sampling period (with no resampling)
    
    '''
    import wirecell.sigproc.response.persist as per
    import wirecell.resp.resample as res

    fr = per.load(frfile)
    if period:
        fr.period = unitify(period)
    if rolloff:
        fr = res.rolloff(fr, rolloff)



    per.dump(output, fr)

@cli.command("resample")
@click.option("-t", "--tick", default=None, type=str,
              help="Resample the field response to have this sample period with units, eg '64*ns'")
@click.option("-e", "--error", default=1e-6,
              help="Allowed error in LMN rationality condition.")
@click.option("-p", "--pad", default="zero",
              type=click.Choice(["zero","linear","first","last"]),
              help="The time domain padding strategy")
@click.option("-o", "--output", default="/dev/stdout",
              help="File in which to write the result")
@click.argument("frfile")
def resample(tick, error, pad, output, frfile):
    '''Resample the FR.

    The initial sampling period Ts (fr.period)) and the resampled period Tr
    (--tick) must satisfy the LMN rationality condition.

    The total duration of the resampled responses may change.

    See also:

    - wirecell-resp condition
    - wirecell-util lmn 
    - wirecell-sigproc fr2npz
    - wirecell.util.lmn.interpolate()
    - LMN resampling paper

    '''

    import wirecell.sigproc.response.persist as per
    import wirecell.resp.resample as res

    tick = unitify(tick)

    fr = per.load(frfile)
    fr = res.resample(fr, tick, eps=error, time_padding=pad)

    per.dump(output, fr)


def zero_centered_freqs_hz(N, T):
    T_sec = T/units.s

    dF = 1/(N*T_sec)
    if N%2:             # odd
        # [-H,...,-1,0,1,...,H]
        H = (N-1)/2
        return numpy.linspace(-H*dF, H*dF, N, endpoint=True) 
    else:               # even, Nyquist bin
        # [-H,...,-1,0,1,...,H,H+1]
        H = N/2 - 1
        return numpy.linspace(-H*dF, (H+1)*dF, N, endpoint=True)
    

@cli.command("compare")
@click.option("--prange", default="0,1,2",
              help="Planes as comma separated indices")
@click.option("--logy/--no-logy", default=False, help="Plot Y axis in log scale")
@click.option("--irange", default='0',
              help="Impact range as comma separated integers")
@click.option("--arange", default=['0:100,-10:10'], multiple=True,
              help="Set time and frequency array ranges in numpy like notation but with time in us and frequency in MHz.")
@click.option("-g", "--gain", default=None, type=str,
              help="Set gain in units, eg '14*mV/fC'.")
@click.option("-s", "--shaping", default=None, type=str,
              help="Set shaping time in units, eg '2*us'.")
@click.option("-o", "--output", default="spec.pdf",
              help="Output plot filename")
@click.argument("responsefile", nargs=-1)
@click.pass_context
def compare(ctx, prange, logy, irange, arange, gain, shaping, output, responsefile):
    '''
    Compare multiple response files.

    If gain/shaping are given convolve the ER.
    '''
    from wirecell.sigproc.response import electronics
    import wirecell.sigproc.response.persist as per
    import wirecell.sigproc.response.plots as plots
    from wirecell.util.plottools import pages, rescaley

    prange = list(map(int, prange.split(',')))
    irange = list(map(int, irange.split(',')))
    aranges = list()
    for one in arange:
        t,f = one.split(',')
        ts = tuple(map(float, t.split(':')))
        fs = tuple(map(float, f.split(':')))
        aranges.append((ts, fs))
    del arange

    gain = unitify(gain)
    shaping = unitify(shaping)

    colors = ["black","red","blue"]
    styles = ["solid","solid","solid"]

    frs = [per.load(rfile) for rfile in responsefile]

    def plot_paths(axes, rfile, pind, n, plane, trange, frange):
        fr = frs[n]
        pr = fr.planes[plane]

        path = pr.paths[pind]
        wave = path.current
        N = wave.size
        T = fr.period

        times = numpy.linspace(0, N*T, N, endpoint=False)

        cspec = numpy.fft.fft(wave)
        if gain and shaping:
            # gain comes with units of [volts]/[charge] but we deal in current
            # so need [volts]/[current] = [volts]/([charge]/[time]).
            cgain = gain * T/units.s
            eresp = electronics(times, cgain, shaping)
            espec = numpy.fft.fft(eresp)
            cspec *= espec
            wave = numpy.real(numpy.fft.ifft(cspec))

        spec = numpy.abs(cspec)
            
        times /= units.us

        spec = numpy.fft.ifftshift(spec)
        max_hz = 1/(T/units.s)
        Fnyq = max_hz / 2

        freqs = zero_centered_freqs_hz(N,T)
        freqs /= 1e6

        args = dict(color=colors[n], linestyle=styles[n], label=rfile, drawstyle='steps')

        axes[0].plot(times, wave, **args)
        axes[0].set_xlim(*trange)
        axes[0].set_xlabel('time [us]')
        axes[0].legend()
        # rescaley(axes[0], times, wave, trange)

        axes[1].plot(freqs, spec, **args)
        axes[1].set_xlim(*frange)
        if logy:
            axes[1].set_yscale('log')
        axes[1].set_xlabel('frequency [MHz]')
        axes[1].legend()
        # rescaley(axes[1], freqs, spec, frange)

        
    with pages(output) as printer:
        for plane in prange:
            for trange, frange in aranges:
                plt.clf()
                fig,axes = plt.subplots(nrows=1, ncols=2)

                for ind in irange:
                    for n, rfile in enumerate(responsefile):
                        plot_paths(axes, rfile, ind, n, plane, trange, frange)

                fig.suptitle(f'plane {plane}, imps:{irange}')
                printer.savefig()

        if gain and shaping:

            plt.clf()
            fig,axes = plt.subplots(nrows=1, ncols=2)

            for n, rfile in enumerate(responsefile):
                fr = frs[n]

                T = fr.period
                N = fr.planes[0].paths[0].current.size

                freqs = zero_centered_freqs_hz(N,T)
                freqs /= 1e6

                times = numpy.linspace(0, N*T, N, endpoint=False)
                eresp = electronics(times, gain, shaping)
                espec = numpy.abs(numpy.fft.fft(eresp))
                espec = numpy.fft.ifftshift(espec)

                times /= units.us

                args = dict(color=colors[n], linestyle=styles[n])

                T_units = T/units.us

                axes[0].plot(times, eresp, **args, label=f'Q: {N=} T={T/units.us} us')
                axes[0].plot(times, eresp*T_units, color=colors[n], linestyle="dashed", label=f'I: {N=} T={T/units.us} us')
                axes[0].set_xlim(0,10)
                #axes[0].set_yscale('log')
                axes[0].set_xlabel('time [us]')

                axes[1].plot(freqs, espec, **args)
                axes[1].plot(freqs, espec*T_units, color=colors[n], linestyle="dashed")
                axes[1].set_xlim(0,0.5/(T/units.us))
                axes[1].set_yscale('log')
                axes[1].set_xlabel('frequency [MHz]')

            fig.legend()
            fig.suptitle('Electronics response')
            plt.tight_layout()
            printer.savefig()



from wirecell.util import lmn
def fr_sig(fr, name=None, impact=66, plane=0):
    '''
    Return one path response as a signal.
    '''
    pr = fr.planes[plane]
    try:
        wave = pr.paths[impact].current
    except IndexError:
        npaths = len(pr.paths)
        nwires = npaths//6
        ntry = int(6*(nwires-1)/2 + 5)
        print(f'No impact {impact} out of {npaths}, probably {nwires} wires, try --impact={ntry}')
        raise
    return lmn.Signal(lmn.Sampling(T=fr.period, N=wave.size), wave=wave, name=name)

def fr_arr(fr, plane=0):
    '''
    Return FR as 2D (nimp,ntick) array.

    If plane is given, return only that plane
    '''
    prs = fr.planes
    if plane is not None:
        prs = [prs[plane]]
    nchan = sum([len(pr.paths) for pr in prs])
    nimps = prs[0].paths[0].current.size
    ret = numpy.zeros((nchan, nimps))
    imp = 0
    for pr in prs:
        for path in pr.paths:
            ret[imp] = path.current
            imp += 1
    return ret


from wirecell.sigproc.response import electronics
def eresp(ss, name="coldelec", gain=14*units.mV/units.fC, shaping=2*units.us):
    '''
    Return electronics response as a signal.
    '''
    times = numpy.linspace(0, ss.N*ss.T, ss.N, endpoint=False)
    eresp = electronics(times, gain, shaping)
    return lmn.Signal(ss, wave = eresp, name=name)

def label(sig):
    '''
    Return a legend label for signal.
    '''
    ss = sig.sampling
    return f'${sig.name},\ N={ss.N},\ T={ss.T/units.ns:.0f}\ ns$'

def plot_signals(sigs, tlim=None, flim=None, tunits="us", funits="MHz",
                 iunits="femtoampere", ilim=None,
                 *args, **kwds):
    '''
    Plot signals in time and frequency domain.

    Any args are dicts matched to signals and passed to their plt.plot()'s.
    Any kwds are are passed to all plt.plot()'s.
    '''
    fig,axes = plt.subplots(nrows=1, ncols=2)

    tunits_v = unitify(tunits)
    funits_v = unitify(funits)
    iunits_v = unitify(iunits)

    colors = ["black","red","blue"]

    for ind, sig in enumerate(sigs):
        try:
            pargs = dict(kwds, args[ind])
        except IndexError:
            pargs = dict(kwds)

        ss = sig.sampling
        wave = sig.wave / iunits_v
        spec = numpy.fft.ifftshift(numpy.abs(sig.spec))
        times = ss.times/tunits_v
        freqs = ss.freqs_zc/funits_v

        pargs.update(label = label(sig),
                     drawstyle='steps',
                     color=colors[ind], linewidth=len(sigs)-ind)
        axes[0].plot(times, wave, **pargs)
        axes[1].plot(freqs, spec, **pargs)

        axes[0].set_xlabel(f'time [{tunits}]')
        axes[0].set_ylabel(f'signal [{iunits}]')
        axes[1].set_xlabel(f'frequency [{funits}]')

    if ilim:
        axes[0].set_ylim(ilim[0]/iunits_v, ilim[1]/iunits_v)
    if tlim:
        axes[0].set_xlim(tlim[0]/tunits_v, tlim[1]/tunits_v)
    if flim:
        axes[1].set_xlim(flim[0]/funits_v, flim[1]/funits_v)
    axes[0].set_title("waveform")
    axes[1].set_title("spectrum")
    axes[1].legend()
    return fig, axes

def plot_ends(arrays, names, iunits='femtoampere'):
    fig,ax = plt.subplots(nrows=1, ncols=1)

    iunits_v = unitify(iunits)

    for ind, arr in enumerate(arrays):
        arr = arr / iunits_v
        col = ['black','red'][ind]
        name = names[ind]
        f = arr[:,0]
        l = arr[:,-1]
        d = l - f
        ax.plot(f, color=col, linestyle='dashed', drawstyle='steps', label=f'${name}$ first')
        ax.plot(l, color=col, linestyle='dotted', drawstyle='steps', label=f'${name}$ last')
        ax.plot(d, color=col, linestyle='solid', drawstyle='steps', label=f'${name}$ diff')

    ax.set_xlabel('impact positions')
    ax.set_ylabel(f'[{iunits}]')
    ax.legend()
    return fig, ax
    

def multiply_period(current, name=None):
    '''
    Return a signal sampling integrated charge from one sampling instantaneous current.
    '''
    ss = current.sampling
    charge = current.wave * ss.T
    return lmn.Signal(ss, wave=charge, name=name)

import wirecell.sigproc.response.persist as per
from wirecell.util.fileio import wirecell_path
def load_fr(detector):
    '''
    Load first FR for a detector
    '''
    paths = wirecell_path()
    got = per.load(detector, paths=paths)
    if isinstance(got, list):
        return got[0]
    return got

@cli.command("lmn-fr-plots")
@click.option("-i", "--impact", default=6*10+5, # on top of central wire of interest
              help="The impact position index in the FR.")
@click.option("-p", "--plane", default=2,
              help="The electrode plane.")
@click.option("-P", "--period", default=None,
              help="Override the sampling time period in the FR.")
@click.option("-S", "--zoom-start", default='70*us',
              help="The start time for 'zoom'.")
@click.option("-W", "--zoom-window", default='20*us',
              help="The time window over which to 'zoom'")
@click.option("-d", "--detector-name", default='uboone',
              help="The canonical detector name")
@click.option("-r", "--field-file", default=None,
              help="Explicit field response file instead of nominal one based on detector name.")
@click.option("-t", "--tick", default='64*ns',
              help="Resample the field response to have this sample period with units, eg '64*ns'")
@click.option("-O", "--org-output", default=None,
              help="Generate an org mode file with macros expanding to figure inclusion.")
@click.option("-o","--output",default="lmn-fr-plots.pdf")
def lmn_fr_plots(impact, plane, period,
                 zoom_start, zoom_window,
                 detector_name, field_file,
                 tick, org_output, output):
    '''
    Make plots for LMN FR presentation.
    '''
    Tr = unitify(tick)

    import wirecell.resp.resample as res
    from wirecell.util.plottools import pages

    FRs = load_fr(field_file or detector_name)
    if period:
        FRs.period = unitify(period)
    sigs = fr_sig(FRs, "I_{og}", impact, plane)
    arrs = fr_arr(FRs)

    FRr = res.resample(FRs, Tr)
    sigr = fr_sig(FRr, "I_{rs}", impact, plane)
    arrr = fr_arr(FRr)

    ers = eresp(sigs.sampling, "E_{og}")
    err = eresp(sigr.sampling, "E_{rs}")

    dsigs = lmn.convolve(sigs, ers, name='I_{og} \otimes E_{og}')
    dsigr = lmn.convolve(sigr, err, name='I_{rs} \otimes E_{rs}')
    dsigr2 = lmn.interpolate(dsigs, Tr, name="(I_{og} \otimes E_{og})_{rs}")

    dtsigs = multiply_period(dsigs, name='T \cdot I_{og} \otimes E_{og}')
    dtsigr = multiply_period(dsigr, name='T \cdot I_{rs} \otimes E_{rs}')
    dtsigr2 = multiply_period(dsigr2, name="T \cdot (I_{og} \otimes E_{og})_{rs}")

    qsigs = multiply_period(sigs, name='Q_{og}')
    qsigr = multiply_period(sigr, name='Q_{rs}')
    qdsigs = lmn.convolve(qsigs, ers, name='Q_{og} \otimes E_{og}')
    qdsigr = lmn.convolve(qsigr, err, name='Q_{rs} \otimes E_{rs}')

    full_range = dict(tlim=(0, sigs.sampling.T*sigs.sampling.N),
                      flim=(0*units.MHz, 10*units.MHz))

    sigs_dur = sigs.sampling.T*sigs.sampling.N
    front_range = dict(tlim=(0, 0.1*sigs_dur),
                      flim=(0*units.MHz, 0.03*units.MHz))
    back_range = dict(tlim=(0.9*sigs_dur, 1.0*sigs_dur),
                      flim=(0*units.MHz, 0.03*units.MHz))

    zoom_start = unitify(zoom_start)
    zoom_window = unitify(zoom_window)
    zoom_range = dict(tlim=(zoom_start, zoom_start+zoom_window),
                      flim=(0*units.MHz, 6*units.MHz))

    conv_range = dict(tlim=(zoom_start, zoom_start+zoom_window),
                      flim=(0*units.MHz, 1*units.MHz))

    orglines = list()
    def orgit(name):
        page = 1+len(orglines)
        pat = r'#+macro: %s \includegraphics[width=\textwidth, page=%s]{%s}'
        text = pat%(name, page, output)
        orglines.append(text)

    with pages(output) as printer:

        def newpage(fig, name, title=''):
            orgit(f'{name}-{detector_name}-{plane}')
            if title:
                fig.suptitle(title)
            fig.tight_layout()
            printer.savefig()
            plt.clf()

        frtit = f'current field response: $FR$ ({detector_name} plane {plane} impact {impact})'

        fig,_ = plot_signals((sigs, sigr), iunits='femtoampere', **full_range)
        newpage(fig, 'fig-fr', frtit)

        fig,_ = plot_signals((sigs, sigr), iunits='femtoampere', **zoom_range)
        newpage(fig, 'fig-fr-zoom', frtit + ' zoom')

        ilim = 0.05*units.femtoampere
        fig,_ = plot_signals((sigs, sigr), iunits='femtoampere', ilim=(-ilim,ilim), **front_range)
        newpage(fig, 'fig-fr-front', frtit)

        fig,_ = plot_ends((arrs, arrr), (sigs.name, sigr.name), iunits='femtoampere')
        newpage(fig, 'fig-ends', 'current response ends')

        fig,_ = plot_signals((ers, err),
                             iunits='mV/fC',
                             tlim=(0*units.us, 20*units.us),
                             flim=(0*units.MHz, 1*units.MHz))
        newpage(fig, 'fig-cer', 'cold electronics response: $ER$')
        
        fig,_ = plot_signals((dsigs, dsigr, dsigr2),
                             iunits='femtoampere*mV/fC',
                             **conv_range)
        newpage(fig, 'fig-fr-er', f'convolution: $FR \otimes ER$ ({detector_name} plane {plane} impact {impact})')


        fig,_ = plot_signals((dtsigs, dtsigr, dtsigr2),
                             iunits='us*femtoampere*mV/fC',
                             **conv_range)
        newpage(fig, 'fig-t-fr-er', f'convolution: $T\cdot FR \otimes ER$ ({detector_name} plane {plane} impact {impact})')


        fig,_ = plot_signals((qsigs, qsigr),
                             iunits='fC',
                             **zoom_range)
        newpage(fig, 'fig-q', f'charge field response $Q = T\cdot FR$ ({detector_name} plane {plane} impact {impact})')

        fig,_ = plot_signals((qdsigs, qdsigr), iunits='mV', **conv_range)
        newpage(fig, 'fig-v', f'voltage response ({detector_name} plane {plane} impact {impact})')

        fig,axes = plot_signals((qdsigs, qdsigr), iunits='mV', **front_range)
        axes[0].set_ylim(-0.5e-6, 0)
        newpage(fig, 'fig-v-front', f'voltage response ({detector_name} plane {plane} impact {impact}, zoom front)')

        fig,axes = plot_signals((qdsigs, qdsigr), iunits='mV', **back_range)
        tiny = 1e-11
        axes[0].set_ylim(-tiny, tiny)
        newpage(fig, 'fig-v-back', f'voltage response ({detector_name} plane {plane} impact {impact}, zoom back)')
        print('q=', 100*numpy.sum(qdsigs.wave) / numpy.sum(numpy.abs(qdsigs.wave)), '%')

    if org_output:
        with open(org_output, "w") as oo:
            oo.write('\n'.join(orglines))
            oo.write('\n')

def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
