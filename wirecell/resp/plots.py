import numpy
import matplotlib.pyplot as plt
from wirecell import units
from wirecell.util import lmn
from wirecell.util.functions import unitify, unitify_parse

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
                 drawstyle='steps',
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

def plot_shift(sig1, sig2,
               tlim=None, flim=None, tunits="us", funits="MHz",
               iunits="femtoampere", ilim=None,
               drawstyle='steps',
               *args, **kwds):
    '''
    Plot the amplitude of the two signals in frequency domain and the phase
    of the ratio of their frequency domain samples.

    The two signals must have identical sampling.

    '''
    fig,axes = plt.subplots(nrows=1, ncols=2)

    tunits_v = unitify(tunits)
    funits_v = unitify(funits)
    iunits_v = unitify(iunits)

    colors = ["black","red","blue"]
    
    

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
    

def plot_wave(ax, sig, pos=False, **kwds):
    y = sig.wave
    if pos:
        y = numpy.abs(sig.wave)
    x = sig.sampling.times / units.us
    ax.plot(x, y, label=f'${sig.name}$', **kwds)
    ax.set_xlabel('time [us]')


def plot_wave_diffs(sigs, primary=0, xlim=None,
                    tunits = units.us, per=(), **kwds):
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

    prim = sigs[primary]
    for sind, sig in enumerate(sigs):
        pkwds = dict(kwds)
        if per:
            pkwds.update(per[sind])

        plot_wave(axes[0], sig, linestyle='', **pkwds)

        if sind == primary:
            continue
        reg = sig.linterp(prim.sampling)
        diff = reg.wave - prim.wave
        lab = f'{reg.name} - {prim.name}'
        axes[1].plot(reg.sampling.times / tunits, diff, label=f'${lab}$', **pkwds)


    axes[0].legend()
    axes[0].set_title('Wave')
    axes[0].set_xlabel('time [us]')

    axes[1].legend()
    axes[1].set_title(f'Difference from ${prim.name}$')
    axes[1].set_xlabel('time [us]')
    axes[1].set_ylabel('difference')
    if xlim:
        xlim = numpy.array(xlim) / tunits
        axes[0].set_xlim(*xlim)
        axes[1].set_xlim(*xlim)
    return fig, axes

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
