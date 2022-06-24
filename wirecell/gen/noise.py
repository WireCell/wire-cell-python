#!/usr/bin/env python3

'''
This has some noise spectrum related code which prototypes what is in
NoiseTools.  See also wirecell.gen.test.test_noise which makes some
diagnostic plots using this module.
'''

import numpy
from math import sqrt, pi, ceil, floor

def rayleigh(x,sigma=1):
    s2 = sigma**2
    return (x/s2)*numpy.exp(-0.5*x**2/s2)

def hermitian_mirror(spec):
    hm = numpy.array(spec)

    # nyquist bin index if size is even, else bin just below Nyquist edge.
    halfsize = hm.size//2

    # zero freq must be real
    hm[0] = numpy.abs(hm[0])
    hm[1:halfsize] = hm[1:halfsize]
    if 0 == hm.size%2:          # even with Nyquist bin
        hm[halfsize] = numpy.abs(hm[halfsize])
        hm[halfsize+1:] = numpy.conjugate(hm[halfsize-1:0:-1])
    else:
        hm[halfsize+1:] = numpy.conjugate(hm[halfsize:0:-1])
    return hm

def fictional(freqs, rel=0.1):
    r = rayleigh(freqs, freqs[-1]*rel)
    return hermitian_mirror(r)

def frequencies(n, period):
    fmax = 1.0/period
    return numpy.linspace(0,fmax,n,False) 

class Collect:
    def __init__(self, nsamples):
        self._lin = numpy.zeros(nsamples);
        self._sqr = numpy.zeros(nsamples);
        self.count = 0;

    @property
    def size(self):
        return self._lin.size

    def add(self, wave):
        wave.resize(self.size)
        spec = numpy.fft.fft(wave)
        mag2 = numpy.abs(spec*numpy.conj(spec))
        self._lin += numpy.sqrt(mag2)
        self._sqr += mag2
        self.count += 1

    @property
    def linear(self):
        return self._lin/self.count

    @property
    def square(self):
        return self._sqr/self.count

    @property
    def energy(self):
        return numpy.sum(self.square)/self.size


class Spec:
    def __init__(self, amp, period):
        '''A spectral amplitude covering full frequency from 0 to
        Nyquest requency of 1/period.  '''
        self.amp = hermitian_mirror(amp)
        self.period = period
        self.freqs = numpy.linspace(0, self.fsample, amp.size, False)

    @property
    def size(self):
        return self.amp.size

    @property
    def half(self):
        '''Return Nyquist bin index (if size is even) or bin just
        below Nyquist frequency edge. '''
        return self.size//2

    @property
    def frayleigh(self):
        return self.fsample/self.size

    @property
    def fnyquist(self):
        return 0.5*self.fsample

    @property
    def fsample(self):
        return 1.0/self.period
    
    def dup(self):
        '''Return copy of self'''
        return Spec(self.amp, self.period)

    @property
    def sigma(self):
        return sqrt(2/pi)*self.amp

    @property
    def energy(self):
        '<|X_k|^2> = 2sigma^2, <|X_k|> = sqrt(pi/2)*sigma'
        return (4/pi)*numpy.sum(self.amp**2) / self.size

    @property
    def random_sigmas(self):
        nhalf = self.half
        if 0 == self.size % 2:
            nhalf += 1
        # convert from |X_k| to sigma_k
        sigmas = sqrt(2/pi)*self.amp[:nhalf]
        r = numpy.random.normal(size=nhalf)*sigmas
        i = numpy.random.normal(size=nhalf)*sigmas
        s = r + i*1j
        s.resize(self.size)
        s = hermitian_mirror(s)
        return s

    @property
    def random_wave(self):
        s = self.random_sigmas
        return numpy.fft.ifft(s).real

    def waves(self, nwaves=None):
        '''Return nwaves random waves'''
        if nwaves is None:
            nwaves = self.size
        return numpy.array([self.random_wave for w in range(nwaves)])
        

    def roundtrip(self, nwaves=None):
        ''' Return a new spec produced by a round-trip through
        generating and collecting waveforms generated from the
        original.'''
        waves = self.waves(nwaves)
        return Spec(numpy.sum(numpy.abs(numpy.fft.fft(waves,axis=1)), axis=0)/waves.shape[0], self.period)

    def time_energy(self, nwaves=None):
        'Sample nwaves and return mean energy'
        return waves_energy(self.waves(nwaves))

    def time_rms(self, nwaves=None):
        'Sample nwaves and return mean RMS'
        return waves_rms(self.waves(nwaves))

    def interp(self, newsize):
        '''Return a new spec interpolated from this to given size

        Result is normalized by sqrt(newsize/oldsize).

        The sample period is held constant.
        '''
        if newsize == self.size:
            return self.dup()
        newfreqs = numpy.linspace(0, self.fsample, newsize, False)
        newamps = numpy.interp(newfreqs, self.freqs, self.amp)
        norm = sqrt(newsize/self.size)
        return Spec(norm*newamps, self.period)

    def interp_fft(self, newsize):
        '''Return a new spec interpolated from this to given size

        Result is normalized by sqrt(newsize/oldsize).

        The sample period is held constant.

        This does a roundtrip FFT with time padding but does not
        really work as we lose phase due to taking abs().

        '''
        if newsize == self.size:
            return self.dup()
        t = numpy.fft.ifft(self.amp)
        t.resize(newsize)
        newamps = numpy.abs(numpy.fft.fft(t))
        newfreqs = numpy.linspace(0, self.fsample, newsize, False)
        norm = sqrt(newsize/self.size)
        return Spec(norm*newamps, self.period)

    def extrap(self, newsize, constant=None):
        '''Return new spec after central extrapolation.

        This is equivalent to an interpolation in time.

        This extrapolation adds more high-frequency bins with a
        constant value.  If constant is None, the existing amplitude
        at the Nyquist frequency is used to fill in the amplitude of
        the new bins.

        The total waveform time is unchanged while sampling and
        Nyquist frequencies increase and thus the period decreases by
        the ratio of new to old sizes.

        The new spectrum is normalized by sqrt(newsize/oldsize) in
        order that energy is conserved.

        '''
        if newsize == self.size:
            return self.dup()
        if newsize < self.size:
            raise ValueError("extrapolation can not reduce size")

        if constant is None:
            constant = self.amp[self.half]
        toadd = newsize - self.size
        newamp = numpy.hstack((self.amp[:self.half],
                               constant*numpy.ones(toadd),
                               self.amp[self.half:]))
        newperiod = self.period*self.size/newsize
        norm = sqrt(newsize/self.size)
        return Spec(norm*newamp, newperiod)

    def alias(self, newsize):
        '''Downsample in time by an alias in frequency to newsize.

        To be exact, newsize must be a factor of oldsize.

        Keeps FRayleigh constant while size decreases so FNyquist
        decreases and period increases.

        This normalizes to sqrt(newsize/oldsize) so that energy is
        conserved in the case that the original spectrum has had an
        anti-alias filter applied such that the aliased bins are
        contain zero amplitude.
        '''

        if newsize == self.size:
            return self.dup()
        # if newsize > self.size:
        #     raise ValueError("alias can not increase size")
        newamp = numpy.zeros(newsize)

        L = ceil(self.size/newsize)
        M = newsize//2

        naliased = 0;
        for m in range(newsize//2 + 1):
            for l in range(L):
                oldind = m + l*M
                if oldind > self.half:
                    break       # effectively zero-padded
                newamp[m] += self.amp[oldind]
                if l > 0 and self.amp[oldind] > 0:
                    naliased += 1

        newamp = hermitian_mirror(newamp)
        newperiod = self.period * self.size/newsize
        # if naliased:
        #     norm = min(1.0, self.period/newperiod)
        # else:
        #     norm = sqrt(newsize/self.size)
        # norm = min(1.0, self.period/newperiod)
        norm = sqrt(newsize/self.size)
        return Spec(norm*newamp, newperiod)

    def resample(self, size, period):
        '''Return spec of given new size and sampling period.

        This is an interpolation followed by an extrapolation or alias
        depending on if new period is smaller or larger than old
        period.
        '''

        # first, want our Frayleigh to match
        # Fr1 = 1/(N1 T1) -> N1 T1 = N2 T2, N2 = N1 T1/T2
        newsize = ceil(size * period/self.period)
        interp = self.interp(newsize)

        if period < self.period:
            ret = interp.extrap(size)
        else:
            ret = interp.alias(size)
        return ret


def gaussian_wave(rms, nsamples):
    return numpy.random.normal(0, rms, size=nsamples)

def gaussian_waves(rms, nsamples, nwaves=None):
    if nwaves is None:
        nwaves = nsamples
    return numpy.array([gaussian_wave(rms,nsamples) for n in range(nwaves)])

def waves_energy(waves):
    'Return mean energy of (nwaves,nsamples) waves'
    return numpy.sum(waves**2)/waves.shape[0]

def waves_rms(waves):
    'Return mean RMS of (nwaves,nsamples) waves'
    return numpy.sum(numpy.sqrt(numpy.sum(waves**2, axis=1)/waves.shape[1]))/waves.shape[0]

def gaussian_spec(rms, nsamples, nwaves=None):
    waves = gaussian_waves(rms, nsamples, nwaves)
    specs = numpy.fft.fft(waves, axis=1)
    amps = numpy.abs(numpy.sqrt(specs*numpy.conj(specs)))
    amp = numpy.sum(amps, axis=0)/nwaves
    mean_time_energy = numpy.sum(waves*numpy.conj(waves))/nwaves
    mean_freq_energy = numpy.sum(numpy.abs(specs*numpy.conj(specs)))/(nwaves*nsamples)
    mean_time_rms = numpy.sum( numpy.sqrt(numpy.abs(waves*numpy.conj(waves))/nsamples) )/nwaves
    return (mean_time_rms, mean_time_energy, mean_freq_energy,amp)

    
