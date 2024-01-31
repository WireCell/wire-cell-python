#!/usr/bin/env python
'''
The LMN resampling method from the paper <TBD>.

fixme: this currently only supports signals in the form of 1D arrays.
'''

import numpy
import math
from numpy import pi
import dataclasses
import matplotlib.pyplot as plt
from wirecell.util.cli import debug

@dataclasses.dataclass
class Sampling:
    '''
    Characterize a discrete sampling of a signal as a finite sequence.
    '''

    T: float
    '''
    The sampling time period
    '''

    N: int
    '''
    The number of samples
    '''

    @property
    def F(self):
        'The maximum sampling frequency'
        return 1.0/self.T

    @property
    def duration(self):
        'The time duration of the signal'
        return self.N*self.T

    @property
    def dF(self):
        'Bandwidth of a frequency sample'
        return 1.0/(self.duration)

    @property
    def nyquist(self):
        'The Nyquist frequency'
        return self.F/2.0

    @property
    def times(self):
        'Sequence of sample times'
        return numpy.linspace(0, self.duration, self.N, endpoint=False)

    @property
    def freqs(self):
        'Sequence of sample frequencies over the "Nyquist-centered" range.'
        return numpy.linspace(0, self.F, self.N, endpoint=False)

    @property
    def freqs_zc(self):
        'Sequence of sample frequencies over the "zero-centered" range.'
        if self.N%2:            # odd
            # -F/2-->|[...n_half...][0][...n_half...]|<--F/2
            return numpy.linspace(-self.nyquist, self.nyquist, self.N)
        else:                   # even
            return numpy.linspace(-self.nyquist+self.dF, self.nyquist, self.N)

    def resampling(self, N):
        '''
        Return a new Sampling with same duration and with N samples.
        '''
        if not isinstance(N, int):
            raise ValueError(f'N must be integer, got "{type(N)}"')
        T = self.duration / N
        return Sampling(T, N)

    def __str__(self):
        return f'N={self.N} T={self.T}'

@dataclasses.dataclass
class Signal:
    '''
    A sampled signal in time and frequency domain sequence representation.
    '''

    sampling: Sampling
    '''
    The time and frequency domain sampling.
    '''

    wave: numpy.ndarray
    '''
    Sequence of time domain samples
    '''

    spec: numpy.ndarray
    '''
    Sequence of frequency domain samples
    '''

    name: str = ""
    '''
    A descriptor of this signal.
    '''

    def __init__(self, sampling, wave=None, spec=None, name=None):
        '''
        Construct a signal from a wave, a spectrum or both.

        If one is missing the other is calcualted
        '''
        self.sampling = sampling
        if wave is None and spec is None:
            raise ValueError("must provide at least one of wave or spec")
        self.wave = wave
        self.spec = spec
        self.name = name or str(sampling)

        if wave is None:
            self.wave = numpy.real(numpy.fft.ifft(self.spec))
        if spec is None:
            self.spec = numpy.fft.fft(self.wave)

    def __str__(self):
        return f'{self.name} {self.sampling}'

    @property
    def time_energy(self):
        '''
        The amount of Parseval energy in the time domain.
        '''
        return numpy.sum( numpy.abs( self.wave * numpy.conj(self.wave) ) )

    @property
    def freq_energy(self):
        '''
        The amount of Parseval energy in the frequency domain.
        '''
        return numpy.sum( numpy.abs( self.spec * numpy.conj(self.spec) ) ) / self.sampling.N


def egcd(a, b, eps=1e-6):
    '''Greated common divisor of floating point values a and b.

    Uses the extended Euclidean algorithm.

    A non-zero error is required if values that are not integer valued.

    '''
    def step(a, b):
        if a < eps:
            return (b, 0, 1)
        else:
            g, x, y = step(b % a, a)
            return (g, y - (b // a) * x, x)
    return step(a,b)[0]



def resize_duration(sam, duration, eps=1e-6):
    '''
    Return a new sampling with the given duration.
    '''
    Ns = sam.N * duration/sam.duration
    if abs(Ns - round(Ns)) > eps:
        raise ValueError(f'Resizing from duration {sam.duration} to {duration} requires non-integer number of samples of {sam.N} to {Ns}')
    Ns = round(newNs)
    return Sampling(sam.T, Ns)


def resize(sig, duration, pad=0, name=None):
    '''Return a new signal of the given duration.

    Note, if duration is not an integer multiple of sig's T, the number
    resulting number of samples will be rounded up.  Check the .duration of the
    result if this matter.

    The signal wave will be truncation or extended with values of pad to fit the
    new duration.
    '''
    ss = sig.sampling
    Nr = math.ceil(ss.N * duration/ss.duration)

    wave = sig.wave.copy()
    wave.resize(Nr)          # in place
    if Nr > ss.N:
        wave[ss.N:] = pad

    rs = Sampling(ss.T, Nr)
    return Signal(rs, wave=wave, name = name or sig.name)
    

def rational_size(Ts, Tr, eps=1e-6):
    '''
    Return a minimum size allowing LMN resampling from period Ts to Tr to be rational.
    '''
    dT = Ts - Tr

    n = dT/egcd(Tr, dT, eps=eps)
    rn = round(n)

    err = abs(n - rn)
    if err > eps:
        raise ValueError(f'no GCD for {Tr=}, {Ts=} within error: {err} > {eps}')

    Ns = rn * Tr / dT
    rNs = round(Ns)
    err = abs(rNs - Ns)
    if err > eps:
        raise ValueError(f'rationality not met for {Tr=}, {Ts=} within error: {err} > {eps}')
    
    return rNs


def rational(sig, Tr, eps=1e-6):
    '''
    Return a new signal like sig but maybe extended to have rational size.
    '''
    Ts = sig.sampling.T
    nrat = rational_size(Ts, Tr, eps)
    Ns = sig.sampling.N
    nrag = Ns % nrat
    if not nrag:
        return sig

    npad = nrat - nrag
    Ns += npad
    cur = sig.wave.copy()
    cur.resize(Ns)

    ss = Sampling(Ts, cur.size)
    sig = Signal(ss, wave=cur)
    return sig


def condition(signal, Tr, eps=1e-6, name=None):
    '''Return a signal that is conditioned to be resampled to period Tr.

    The wave will be extended to accomodate a half-cosine with a frequency that
    of the peak of the input spectrum.  It is further padded so that the entire
    waveform is a multiple of the minimimum size for optimal LMN resampling of
    the signal to period Tr.  Any additional samples are given the amplitude
    matching the first sample.
    '''
    Ts = signal.sampling.T
    Ns = signal.sampling.N

    first = signal.wave[0]
    last = signal.wave[-1]
    ipeak = numpy.argmax(numpy.abs(signal.spec))
    fpeak = signal.sampling.freqs[ipeak]

    # Number of time bins needed for a half-cosine at fpeak.
    nsmooth = int(numpy.ceil(1/(2*Ts*fpeak)))
    #print(f'{fpeak=} {Ts=} {nsmooth=}')

    # need at least this much to fit in smoothing half-cosine.
    nmin_needed = Ns + nsmooth

    # Final size must be a multiple of this.
    nmin = rational_size(Ts, Tr)

    # total padded size to reach next multiple of nmin
    npadded = int(numpy.ceil(nmin_needed/nmin)*nmin)
    # print(f'{nmin=} {nmin_needed=} {npadded=}')

    wave = numpy.zeros(npadded, dtype=signal.wave.dtype)
    wave[:Ns] = signal.wave
    wave[Ns:] = first

    # half-cosine parameters
    amp = 0.5*(last - first)
    bl = 0.5*(first + last)
    # the times for making smoothing half-cosine
    smooth_time = numpy.linspace(0, nsmooth*Ts, nsmooth, endpoint=False)
    smoother = bl + amp*numpy.cos(2*pi*fpeak*smooth_time)
    wave[Ns: Ns+nsmooth] = smoother
    return Signal(Sampling(T=Ts, N=npadded), wave=wave, name=name)

def resample(signal, Nr, name=None):
    '''
    Return a new signal of same duration that is resampled to have number of samples Nr.
    '''
    Ns = signal.sampling.N
    resampling = signal.sampling.resampling(Nr)
    Tr = resampling.T

    S = signal.spec
    R = numpy.zeros(Nr, dtype=S.dtype)

    # Number of unique bins in teh half spectrum ignoring the "DC bin" and
    # ignoring the possible Nyquist bin.
    if Ns%2:                    # odd, no nyquist bin
        Ns_half = (Ns-1)//2
    else:                       # even, exclude nyquist bin
        Ns_half = (Ns-2)//2
    if Nr%2:                    # odd, no nyquist bin
        Nr_half = (Nr-1)//2
    else:                       # even, exclude nyquist bin
        Nr_half = (Nr-2)//2

    # Get "half spectrum" size for the non-zero part of resampled spectrum.
    if Nr > Ns:                 # upsample
        n_half = Ns_half
    else:                       # downsample
        n_half = Nr_half        

    # copy DC term and "positive" aka "low-frequency" half spectrum.
    R[:1+n_half] = S[:1+n_half]
    # copy the "negative" aka "high-frequency" half spectrum
    R[-n_half:] = S[-n_half:]

    #print(f'{R.shape=} {Nr_half=} {S.shape=} {Ns_half=} {n_half=}')

    deal_with_nyquist_bin = False
    if deal_with_nyquist_bin:
        if Nr > Ns and not Ns%2: # upsampling from spectrum with Nyquist bin
            val = S[Ns//2]
            R[1+n_half+1] = 0.5*val
            R[-n_half-1]  = 0.5*val
        if Nr < Ns and not Nr%2: # downsampling to spectrum with Nyquist bin
            R[1+n_half+1] = abs(S[1+n_half+1])

    return Signal(resampling, spec = R, name=name)

def interpolate(sig, Tr, eps=1e-6, name=None):
    '''
    Interpolation resampling.

    This meets the rational contraint.
    '''

    rat = rational(sig, Tr, eps)
    # debug(f'interpolate: rationalize {sig.sampling} -> {rat.sampling}')

    Nr = rat.sampling.duration / Tr
    if abs(Nr - round(Nr)) > eps:
        raise LogicError('rational resize is not rational')
    Nr = round(Nr)

    res = resample(rat, Nr)
    # debug(f'interpolate: resample {rat.sampling} -> {res.sampling}')

    # rez = resize(res, sig.sampling.duration)
    rez = res

    # The response is instantaneous current and thus we use interpolation
    # normalization.
    # norm = res.sampling.N / sig.sampling.N
    norm = res.sampling.N / rat.sampling.N

    fin = norm * rez.wave
    return Signal(Sampling(T=Tr, N=fin.size), wave=fin, name=name)


def convolve(s1, s2, mode='full', name=None):
    '''
    Return new signal that is the convolution of the two.

    Mode is as taken by numpy.convolve which provides the kernel.
    '''
    if s1.sampling.T != s2.sampling.T:
        raise ValueError("can not convolve signals if differing sample period")
    wave = numpy.convolve(s1.wave, s2.wave, mode)
    return Signal(Sampling(T=s1.sampling.T, N=wave.size), wave=wave, name=name)
