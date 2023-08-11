#!/usr/bin/env python
'''
Implement simple exercise of the ForWaRD technique.
'''
from math import sqrt, pi
import numpy
from numpy.fft import fft, ifft
import functools
from wirecell import units
from wirecell.util import jsio
from wirecell.util.paths import resolve
from wirecell.sigproc.response import electronics
from wirecell.sigproc.response.persist import load as load_fr
import numbers
from collections import namedtuple

import matplotlib.pyplot as plt

def make_times(N, T, t0=0):
    'Return sequence of times'
    return numpy.linspace(t0, N*T+t0, N, endpoint=False)

def make_freqs(N, T):
    'Return sequence of frequencies'
    return numpy.linspace(0, 1/T, N, endpoint=False)


# Fixme: these details really should be in a json/jsonnet file in
# wire-cell-data!  https://github.com/WireCell/wire-cell-data/issues/6
known_responses = dict(
    uboone=dict(fields="ub-10-half.json.bz2",
                noise="microboone-noise-spectra-v2.json.bz2",
                elec_type="cold",
                peak_gain=14*units.mV/units.fC,
                shaping=2.0*units.us),
    pdps=dict(fields="dune-garfield-1d565.json.bz2",
              noise="protodune-noise-spectra-v1.json.bz2",
              elec_type="cold",
              peak_gain=14*units.mV/units.fC,
              shaping=2.0*units.us),
)

def convert_spec(times, nsamples, period, freqs, amps, const=None, **ignored):
    '''Return spectrum over uniform sampled times.

    Freqs and amps are a irregular sampling of a half spectrum
    originaly sampled with period and nsamples.

    '''
    N = times.size
    T = times[1] - times[0]

    fmax_want = 1.0/T
    freqs_want = make_freqs(N, T)

    scale = sqrt(N / nsamples)

    freqs = numpy.array(freqs)
    amps = numpy.array(amps) * scale

    spec = numpy.interp(freqs_want, freqs, amps)

    # hermitian symmetry
    if N%2:                     # odd
        ncomp = (N-1)//2
        spec[1+ncomp:] = numpy.flip(spec[1:ncomp+1])
    else:                       # even, with nyquist bin
        ncomp = N//2 - 1
        spec[2+ncomp:] = numpy.flip(spec[1:ncomp+1])

    if const is not None:
        const *= scale
        spec = numpy.sqrt(spec*spec + const*const)

    return spec


class DisplayUnit:
    '''
    Convert from WCT system-of-units to display units.
    '''
    kinds = ('time', 'frequency', 'voltage', 'charge', 'voltage/charge', 'count', 'arbitrary')
    

    def __init__(self, name, kind, value):
        '''
        Carry a specific unit conversion from native system of units.

        - name :: string as may be used in an axis: "[unit]".

        - kind :: an element of .kinds

        - value :: display unit value expressed in WCT system of units: units.microsecond.

        '''
        if not kind in self.kinds:
            raise ValueError(f'unknown kind: "{kind}"')
        self.name = name
        self.kind = kind
        self.value = value

    def __call__(self, x):
        '''
        Convert x from WCT system-of-units to display units.
        '''
        return x/self.value

# Collect units for a value in time or frequency space.
# The "name" describes the value, eg "Field response", "Ionization signal"
DisplayUnitsVTF = namedtuple("DisplayUnitsVTF", "name value time freq")

def unit_info():
    '''
    Return dict of various DisplayUnitsVTF expected by plotting functions
    '''
    # Specific display units
    time = units.us
    freq = units.megahertz
    voltage = units.millivolt
    charge = units.femtocoulomb

    us = DisplayUnit("us", "time", time)
    MHz = DisplayUnit("MHz", "frequency", freq)
    fCe = DisplayUnit("fC/e", "charge", charge)
    mV = DisplayUnit("mV", "voltage", voltage)
    mVe = DisplayUnit("mV/e", "voltage", voltage)
    mVfC = DisplayUnit("mV/fC", "voltage/charge", voltage/charge)
    ke = DisplayUnit("kNe", "count", 1000)
    arb = DisplayUnit("arb", "arbitrary", 1)
    return dict(field = DisplayUnitsVTF("Field response", fCe, us, MHz),
                elec = DisplayUnitsVTF("Elec resposne", mVfC, us, MHz),
                det = DisplayUnitsVTF("Det response", mVe, us, MHz),
                noise = DisplayUnitsVTF("Noise", mV, us, MHz),
                ionized = DisplayUnitsVTF("Ionized electrons", ke, us, MHz),
                diffusion = DisplayUnitsVTF("Diffusion", arb, us, MHz),
                drifted = DisplayUnitsVTF("Drifted electrons", ke, us, MHz),
                quiet = DisplayUnitsVTF("Quiet", mV, us, MHz), # no noise, get it? har har har
                )




def wave_energy(wave):
    '''
    Return the Parseval energy in a signal's waveform.
    '''
    return numpy.sum(numpy.abs(numpy.conj(wave)*wave))

def spec_energy(spec):
    '''
    Return the Parseval energy in a signal's spectrum.
    '''
    return numpy.sum(numpy.abs(numpy.conj(spec)*spec)) / len(spec)

class Signal:
    '''
    Model a value sampled uniformly over time.
    '''
    def __init__(self, name, times, values=None):
        '''
        Create a named signal with uniform sample times, values and units.

        The values will be forced same size as times.  Null values are set to a zero array.
        '''
        self.name = name
        self.times = times
        self.T = times[1] - times[0]
        if values is None:
            values = numpy.zeros(times.size)
        elif values.size != times.size:
            values = numpy.copy(values)
            values.resize(times.size)
        self._values = values


    def __str__(self):
        wave = self.waveform
        wmin = numpy.min(wave)
        wmax = numpy.max(wave)
        wsum = numpy.sum(wave)
        t_us = self.T/units.us
        fmax_mhz = 1/t_us
        fmin_khz = fmax_mhz/self.N * 1000
        return f'<Signal: "{self.name}" N={self.N} T={t_us}us F=[{fmin_khz:.2f}kHz,{fmax_mhz:.2f}MHz] wave: mm=[{wmin:.2e},{wmax:.2e}], tot={wsum:.2e}, E={self.wave_energy:.2e}>'

    def __len__(self):
        return len(self.times)

    def _iarith(self, other):
        if isinstance(other, numbers.Number):
            return other
        if isinstance(other, Signal):
            other = other.waveform
        if isinstance(other, numpy.ndarray):
            a = len(self)
            b = len(other)
            if a == b:
                return other
            raise ValueError(f'Signal size mismatch {a} != {b}')
        raise TypeError(f'Not compatible with Signal: {type(other)}')

    def __add__(self, other):
        other = self._iarith(other)        
        return Signal(self.name, self.times, self.waveform + other)
    def __sub__(self, other):
        other = self._iarith(other)        
        return Signal(self.name, self.times, self.waveform - other)
    def __mul__(self, other):
        other = self._iarith(other)        
        return Signal(self.name, self.times, self.waveform * other)

    def __iadd__(self, other):
        other = self._iarith(other)
        self._values += other
        return self

    def __isub__(self, other):
        other = self._iarith(other)
        self._values -= other
        return self

    def __imul__(self, other):
        other = self._iarith(other)
        self._values *= other
        return self

    @property
    def waveform(self):
        return self._values

    @waveform.setter
    def waveform(self, value):
        w = getattr(value, "waveform", None)
        if w is not None:       # Signal-like
            self._values = w
            return
        if isinstance(value, numpy.ndarray):
            self._values = value
            return
        raise TypeError(f'Not compatible with Signal: {type(value)}')            

    @property
    def freqs(self):
        return make_freqs(self.N, self.T)

    @property
    def freqs_hz(self):
        return self.freqs * units.second # double division

    @property
    def N(self):
        return self.times.size

    @property
    def sampling_hz(self):
        return 1/(self.T/units.second)

    @property
    def spectrum(self):
        '''
        Complex representation of waveform in frequency.
        '''
        return fft(self.waveform)

    @property
    def amplitude(self):
        '''
        The real valued spectral amplitude.
        '''
        return numpy.abs(self.spectrum)

    @property
    def wave_energy(self):
        '''
        Return the Parseval energy in a signal's waveform.
        '''
        wave = self.waveform
        return numpy.sum(numpy.abs(numpy.conj(wave)*wave))

    @property
    def spec_energy(self):
        '''
        Return the Parseval energy in a signal's spectrum.
        '''
        spec = self.spectrum
        return numpy.sum(numpy.abs(numpy.conj(spec)*spec)) / self.N

def listify_signals(sigs):
    '''
    Return input as list of Signal objects.
    '''
    if isinstance(sigs, Signal):
        return (sigs,)
    return sigs


class Noise(Signal):
    '''
    A generative noise signal with spectrum from data files.
    '''
    def __init__(self, name, times, detector='uboone', plane=0):
        '''
        Noise spectrum for given detector and plane.
        '''
        super().__init__(name, times)

        nfile = known_responses[detector]['noise']
        jspecs = jsio.load(str(resolve(nfile)))
        for jspec in jspecs:
            if plane != jspec.get('plane',None):
                continue
            self._amplitude = convert_spec(times, **jspec)
            break

    @property
    def spectrum(self):
        return self._amplitude + 1j*numpy.zeros_like(self._amplitude)

    @property
    def amplitude(self):
        '''
        The real valued spectral amplitude.
        '''
        return self._amplitude

    @property
    def waveform(self):
        '''
        Return a fluctuated version of self assuming self is mean noise.
        '''
        N = self.N
        gspec = numpy.zeros(N, dtype='complex')

        # convert from mean to sigma mode parameter
        mean = self.amplitude
        mode = sqrt(2/pi) * mean

        normals = numpy.random.normal(size=N)
        if N%2:                     # odd, only [0] is real
            # need 1 real DC and 2 * (N-1)/2 complex
            ncomp = N//2            # odd division
            gspec[0] = normals[-1]
            comp = normals[0:ncomp] + 1j*normals[ncomp:2*ncomp];
            gspec[1:ncomp+1] = comp
            gspec[ncomp+1:] = numpy.conj(numpy.flip(comp))
        else:                       # even, [0] and [N/2] are real
            ncomp = N//2-1
            gspec[0] = normals[-1]
            comp = normals[0:ncomp] + 1j*normals[ncomp:2*ncomp]
            gspec[1:ncomp+1] = comp
            gspec[ncomp+1] = normals[-2] # nyquist bin
            gspec[ncomp+2:] = numpy.conj(numpy.flip(comp))

        return numpy.real(ifft(gspec*mode))


class Convolution(Signal):
    '''
    A signal formed from the convolution of two signals.
    '''
    def __init__(self, name, one, two, mode="full"):
        '''
        Create measure from two signals.
        '''
        wave = numpy.convolve(one.waveform, two.waveform, mode)
        wave.resize(len(one.waveform))
        super().__init__(name, one.times, wave)

        self.one = one
        self.two = two

    def component(self, key):
        '''
        Return component by identifying key.
        '''
        if key in ("", "both", "whole", "self", 0, 3):
            return self
        if key in ("one", 1, self.one.name):
            return self.one
        if key in ("two", 2, self.two.name):
            return self.two
        raise KeyError(f'no matching component for key "{key}"')



#
# Special constructions
# 


class FieldResponse(Signal):
    '''Single-electron impulse field response signal'''

    def __init__(self, name, detector='uboone', plane=0, impact=None):
        '''
        If impact is None, an average is taken over the first wire
        region else it may provide an index into FR's paths array

        Note, .times will be that from the data file.

        '''
        frfile = known_responses[detector]['fields']
        frfile = str(resolve(frfile))
        frs = load_fr(frfile)

        impacts = [impact]
        if impact is None:
            impacts = list(range(6))

        pr = frs.planes[plane]
        res = numpy.zeros_like(pr.paths[0].current)
        for path in impacts:
            res += pr.paths[path].current
        res = res/len(impacts)
        res *= frs.period       # convert from current to charge in sample

        N = len(res)
        times = make_times(N, frs.period)
        super().__init__(name, times, res)
        self.plane = plane
        self.impact = impact


class ElecResponse(Signal):
    '''
    Electronics impulse response signal.
    '''
    def __init__(self, name, times, detector='uboone'):
        from wirecell.sigproc.response import electronics
        d = known_responses[detector]
        d = {k:v for k,v in d.items() if k in ("peak_gain", "shaping", "elec_type")}
        wave = electronics(times, **d)
        super().__init__(name, times, wave)
        self.__dict__.update(d)


class DetectorResponse(Convolution):
    '''
    Return full (FR x ER) detector impulse response signal.
    '''
    def __init__(self, name, detector = 'uboone'):
        fr = FieldResponse("field", detector)
        er = ElecResponse("elec", fr.times, detector)
        super().__init__(name, fr, er, "full")
        self.fr = fr
        self.er = er


def square(times, width=1*units.us, shift=None, name="square"):
    '''Return a unit square wave signal of width and time shift.

    If shift is None, square is centered.  Else, shift gives delay
    between zero and lower edge.

    Signal name will be "square".
    '''
    N = len(times)
    T = times[1]-times[0]

    wave = numpy.zeros(N)
    nwidth = int(width/T)

    if shift  is None:
        nshift = int(N - nwidth)//2
    else:
        nshift = int(shift/T)

    wave[nshift:nwidth+nshift] = 1
    return Signal(name, times, wave)


#
# functions taking signals
# 


def rebin(signal, nbins, name=None):
    '''Return a new signal with waveform rebinned by summing over nbins.

    New signal will be signal.size//nbins long and same name as
    original.  Times will be every nbins from signal.times.

    '''
    name = name or signal.name

    N = signal.N
    Nfit = N - (N%nbins)
    wave = numpy.copy(signal.waveform)
    wave.resize(Nfit)
    wave = numpy.sum(wave.reshape((-1, nbins)), axis=1)
    newtimes = signal.times[0:Nfit:nbins]
    newsig = Signal(name, newtimes, wave)
    return newsig


def interp_linear(signal, times, renorm=False):
    '''Return signal with waveform values interpolated to new times.

    If renorm is True, new signal will have same Parseval energy as old.
    '''
    wave = numpy.interp(times, signal.times, signal.waveform)
    newsig = Signal(signal.name, times, wave)
    if renorm:
        newsig.waveform *= signal.wave_energy / newsig.wave_energy
    return newsig



#
# free function
# 

def gauss_wave(times, sigma=1, tpeak=0, constant=1, name=None):
    '''Return a Gaussian shaped waveform.

    - times :: 1D regular grid array of times.
    - sigma :: optional width of the Gaussian, def=1.0 (in units of time).
    - tpeak :: optional peak location), def=0 (in units of time).
    - constant :: optional amplitude constant, def=1.

    If "tpeak" is None, Gaussian is centered on times. 

    Any of sigma, tpeak and constant may be arrays and if so must be equal length.

    '''
    N = times.size

    def wash(x):
        if isinstance(x, numpy.ndarray):
            return x.reshape((-1,1))
        return x

    sigma = wash(sigma)
    tpeak = wash(tpeak)
    constant = wash(constant)

    if tpeak is None:
        tpeak = times[N//2]

    x = (times-tpeak)/sigma
    g = numpy.exp(-0.5*x*x)
    if len(g.shape) == 1:
        norm = numpy.sum(g)
    else:
        norm = numpy.sum(g, axis=1).reshape((-1,1))

    return constant*g/norm

def gauss_signal(times, sigma=1, tpeak=0, constant=1, name="gauss"):
    return Signal(name, times, gauss_wave(times, sigma, tpeak, constant))

def randgauss_wave(times, n=10,
                   amprange=(1,2),
                   sigrange=(1*units.us, 2*units.us),
                   timerange=(1*units.us, 2*units.us)):
    '''
    Return n x times.size array with rows holding random Gaussian curves.
    '''
    t = numpy.random.uniform(*timerange, n).reshape(-1,1)
    s = numpy.random.uniform(*sigrange, n).reshape(-1,1)
    c = numpy.random.uniform(*amprange, n).reshape(-1,1)
    ret = gauss_wave(times, s, t, c)
    return ret


#
# Plotting related
# 

def plot_legend(ax, label):
    '''
    Add a line to legend with no associated plot
    '''
    ax.plot([],[], ' ', label=label)

def signal_legend(ax, sig, u):
    '''
    Add text to legend for signal.
    '''
    T = u.time(sig.T)
    uT = u.time.name
    F = sig.sampling_hz/1e6
    f = sig.sampling_hz/1e3/sig.N

    plot_legend(ax, f'N={sig.N}, T={T} {uT}')
    plot_legend(ax, f'F=[{f:.2f} kHz, {F} MHz]')
    

def add_spec_to_legend(ax, n, t):
    '''Add lines to legend to show Fourier params relevant to sample
    number and period.

    '''

    t_us = t/units.us
    f_samp_hz = 1.0/(t/units.second)
    f_nyq_mhz = 0.5e-6*f_samp_hz
    f_min_khz = 1e-3*f_samp_hz/n

    plot_legend(ax, f'N={n}, T={t_us} us')
    plot_legend(ax, f'F_min={f_min_khz:.1f} kHz')
    plot_legend(ax, f'F_nyquist={f_nyq_mhz:.1f} MHz')



uvalab = dict(elec=dict(val=units.mV/units.fC, lab="mV/fC"),
              field=dict(val=units.fC, lab="fC/electron"),
              det=dict(val=units.mV, lab="mV/electron"))

def plot_resp_wave(dr, which=('elec','field','det'), roundtrip=False):
    '''
    Plot one or more impact responses from given DetectorResponse.

    If roundtrip, then also plot the ifft() of the corresponding spectrum.

    It is assumed that caller saves the figure.
    '''

    t_us = dr.times/units.us
    nt = len(t_us)

    for name in which:
        uval = uvalab[name]["val"]
        ulab = uvalab[name]["lab"]

        wave = dr.component(name)
        vmax = numpy.max(wave)
        plt.plot(t_us, wave/vmax, label=f'{name} ({vmax/uval:.2e} {ulab})')
        if roundtrip:
            wave = ifft(dr.spec(name))
            plt.plot(t_us, wave[:nt]/vmax, label=f'{name} ({vmax:.2e}, rt)')

    add_spec_to_legend(plt, len(t_us), dr.fr.period)
    plt.legend()
    plt.title(f'Impulse response waveform ({dr.detector})')
    plt.xlabel('time [us]')
    plt.ylabel('(arb)')


def plot_resp_spec(dr, which=('elec','field','det'), max_hz=None):
    '''
    Plot one or more impact response spectra
    '''
    f_mhz = dr.spec_freqs_hz / 1e6
    n = len(f_mhz)
    if max_hz:
        n = numpy.sum(f_mhz <= max_hz/1e6)

    for name in which:
        uval = uvalab[name]["val"]
        ulab = uvalab[name]["lab"]

        spec = numpy.abs(dr.spec(name))
        vmax = numpy.max(spec)
        plt.plot(f_mhz[:n], spec[:n]/vmax, label=f'{name} ({vmax/uval:.2e} {ulab})')

    add_spec_to_legend(plt, len(f_mhz), dr.fr.period)
    plt.legend()

    plt.title(f'Impulse response spectra ({dr.detector})')
    plt.xlabel('Frequency [MHz]')
    
    
def plot_noise(noi, unit, max_freq=1*units.megahertz):
    fig, (sax,wax) = plt.subplots(nrows=2)

    mean = numpy.zeros_like(noi.amplitude)
    nwave = 100
    Ewave = 0
    Espec = 0
    for wind in range(nwave):
        one = Signal('noise', noi.times, noi.waveform) # generates
        mean += one.amplitude
        Ewave += one.wave_energy
        Espec += one.spec_energy
        wax.plot(unit.time(noi.times), unit.value(one.waveform))

    Ewave /= nwave
    Ewave = unit.value(unit.value(Ewave))
    plot_legend(wax, f'energy={Ewave:.1f}{unit.value.name}^2 (N={nwave})')

    Espec /= nwave
    Espec = unit.value(unit.value(Espec))
    
    mean /= nwave
    fbins = noi.freqs <= max_freq

    sax.plot(unit.freq(noi.freqs[fbins]), unit.value(mean[fbins]), label=f'reco (N={nwave})')

    Espec2 = spec_energy(mean)
    Espec2 = unit.value(unit.value(Espec2))
    plot_legend(sax, f'mean E={Espec:.1f}{unit.value.name}^2')
    plot_legend(sax, f'E mean={Espec2:.1f}{unit.value.name}^2')
    plot_legend(sax, f'ratio: {Espec/Espec2:.3f}')

    sax.plot(unit.freq(noi.freqs[fbins]), unit.value(noi.amplitude[fbins]), label='model')

    Espec3 = spec_energy(noi.amplitude)
    Espec3 = unit.value(unit.value(Espec3))
    plot_legend(sax, f'energy={Espec3:.1f}{unit.value.name}^2')

    sax.set_title(f'{unit.name} spectra')
    sax.set_xlabel(f'frequency [{unit.freq.name}]')
    sax.set_ylabel(f'amplitude [{unit.value.name}]')

    wax.set_title(f'{unit.name} waveforms')
    wax.set_xlabel(f'time [{unit.time.name}]')
    wax.set_ylabel(f'amplitude [{unit.value.name}]')

    sax.legend(loc='upper right')
    wax.legend()

    fig.tight_layout()
    return fig
    


def plot_convo(ua, ub, uc, a, bs, max_freq=1*units.megahertz):
    '''Plot a and bs waveform wave/spec of the convolution'''

    fig, ((aax,bax),(wax,sax)) = plt.subplots(nrows=2, ncols=2)

    aax.set_title(f"{ua.name} waveform")
    aax.set_xlabel(f"time [{ua.time.name}]")
    aax.set_ylabel(f"amplitude [{ua.value.name}]")

    bax.set_title(f"{ub.name} waveforms")
    bax.set_xlabel(f"time [{ub.time.name}]")
    bax.set_ylabel(f"amplitude [{ub.value.name}]")

    wax.set_title(f"{uc.name} waveform")
    wax.set_xlabel(f"time [{uc.time.name}]")
    wax.set_ylabel(f"amplitude [{uc.value.name}]")

    sax.set_title(f"{uc.name} spectra")
    sax.set_xlabel(f"frequency [{uc.freq.name}]")
    sax.set_ylabel(f"amplitude [{uc.value.name}]")

    aax.plot(ua.time(a.times), ua.value(a.waveform))
    signal_legend(aax, a, ua)

    for b in bs:
        bax.plot(ub.time(b.times), ub.value(b.waveform))
        c = Convolution(a.name, a, b)

        wax.plot(uc.time(c.times), uc.value(c.amplitude))
        print(max_freq)
        bins = c.freqs <= max_freq
        sax.plot(uc.freq(c.freqs[bins]), uc.value(c.waveform[bins]))

    signal_legend(bax, b, ub)
    signal_legend(wax, c, uc)
        
    aax.legend()
    bax.legend()
    # sax.legend()
    wax.legend()
    fig.tight_layout()



def default_limiter(x,y):
    return x,y


def range_limiter(xmin=0, xmax=None):
    '''
    Return function to limit x,y in given x-axis range
    '''
    def limiter(x,y):
        bins = x >= xmin
        if xmax is not None:
            bins = numpy.logical_and(bins, x <= xmax)
        return x[bins], y[bins]
    return limiter


def plot_waveform(ax, sigs, dunits, xyselect=default_limiter, legend=False):
    '''
    Plot one or more signal wavefors on same axis.

    - ax :: a matplotlib axis
    - sigs :: a Signal object or list of Signal objects
    - dunits :: a DisplayUnitsVTF
    - xyselect :: function(xarr,yarr) -> xarr,yarr to filter plot arrays

    '''
    sigs = listify_signals(sigs)

    titlab = 'waveforms'
    xunits = dunits.time
    yunits = dunits.value

    ax.set_title(f'{dunits.name} {titlab}')
    ax.set_xlabel(f'time [{xunits.name}]')
    ax.set_ylabel(f'amplitude [{yunits.name}]')

    for sig in sigs:
        xarr,yarr = xyselect(sig.times, sig.waveform)
        xarr = xunits(xarr)
        yarr = yunits(yarr)

        label = sig.name if legend else None
        ax.plot(xarr, yarr, label=label)
    if legend:
        ax.legend()


def plot_spectrum(ax, sigs, dunits, xyselect=default_limiter, legend=False):
    '''
    Plot one or more signal spectra on same axis.

    '''
    sigs = listify_signals(sigs)

    titlab = 'spectra'
    xunits = dunits.freq
    yunits = dunits.value

    ax.set_title(f'{dunits.name} {titlab}')
    ax.set_xlabel(f'frequency [{xunits.name}]')
    ax.set_ylabel(f'amplitude [{yunits.name}]')

    for sig in sigs:
        xarr,yarr = xyselect(sig.freqs, sig.amplitude)
        xarr = xunits(xarr)
        yarr = yunits(yarr)
        label = sig.name if legend else None
        ax.plot(xarr, yarr, label=label)
    if legend:
        ax.legend()

def plot_convo(sigs, resps, convos,
               sunits, runits, cunits,
               tlimiter=default_limiter, flimiter=default_limiter):
    '''
    2x2 plot two sets of signals and their mutual convolutions.
    '''
    fig, ((sigax,resax),(wax,sax)) = plt.subplots(nrows=2, ncols=2)

    sigs = listify_signals(sigs)
    resps = listify_signals(resps)
    convos = listify_signals(convos)

    plot_waveform(sigax, sigs, sunits, legend=True)
    plot_waveform(resax, resps, runits, legend=True)
    
    plot_waveform(wax, convos, cunits)
    plot_spectrum(sax, convos, cunits, flimiter)

    signal_legend(sax, convos[0], cunits)
    sax.legend()

    fig.tight_layout()
    return fig

def plot_signal_noise(sig, noi, tot, dunits,
                      tlimiter=default_limiter, flimiter=default_limiter):
    '''
    Make signal+noise plots.
    '''
    fig, (wax,sax) = plt.subplots(nrows=2)
    
    nwave = noi.waveform

    plot_waveform(wax, tot, dunits, tlimiter, legend=True)
    plot_waveform(wax, sig, dunits, tlimiter, legend=True)

    plot_spectrum(sax, tot, dunits, flimiter, legend=True)
    plot_spectrum(sax, noi, dunits, flimiter, legend=True)
    plot_spectrum(sax, sig, dunits, flimiter, legend=True)

    fig.tight_layout()
