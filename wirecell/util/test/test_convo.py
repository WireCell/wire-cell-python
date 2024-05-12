#!/usr/bin/env python

from wirecell import units
from wirecell.util import lmn
from wirecell.util.plottools import pages
from wirecell.resp.plots import plot_signals, multiply_period

import numpy
import matplotlib.pyplot as plt


def gauss(x, mu, sigma, A=1.0):
    return A*numpy.exp(- (x-mu)**2/(2*sigma**2))/(2*sigma)


def sig(T, mu, sigma, A=1.0, name=None, N=None):
    if N is None:
        duration = mu + 10 * sigma
        N = round(duration/T)
    else:
        duration = T*N
    t = numpy.linspace(0, duration, N, endpoint=False)
    return lmn.Signal(lmn.Sampling(T=T, N=N),
                      wave=gauss(t, mu, sigma, A), name=name)


g1 = dict(mu=10*units.us, sigma=2*units.us)
g2 = dict(mu=20*units.us, sigma=3*units.us)

Tfast = 100*units.ns
Ffs = sig(Tfast, name='Ffs', **g1)
Ffl = sig(Tfast, name='Ffl', **g1, N=3*Ffs.sampling.N)

Efs = sig(Tfast, name='Efs', **g2)

Tslow = 500*units.ns
Fss = sig(Tslow, name='Fss', **g1)
Ess = sig(Tslow, name='Ess', **g2)
Nus = int(Tslow/Tfast * Ess.sampling.N)
#Eus = lmn.resample(Ess, Nus, name='Eus').multiply(Nus/Ess.sampling.N)
Eus = lmn.interpolate(Ess, Tfast, name='Eus')

Fs = (Fss, Ffs, Ffl, Ffs)
Es = (Ess, Efs, Efs, Eus)


def tconvolve(a, b, name=None):
    '''
    Convolve in time domain.
    '''
    Ta = a.sampling.T
    Tb = b.sampling.T
    assert (Ta == Tb)

    Na = a.sampling.N
    Nb = b.sampling.N
    N = Na + Nb

    wa = numpy.zeros(N, dtype=a.wave.dtype)
    wb = numpy.zeros(N, dtype=b.wave.dtype)
    cab = numpy.zeros(N, dtype=b.wave.dtype)
    wa[:Na] = a.wave
    wb[:Nb] = b.wave

    for outer in range(N):
        v = 0
        for inner in range(N):
            ind = outer-inner
            v += wa[ind%N] * wb[inner]
        cab[outer] = v

    return lmn.Signal(lmn.Sampling(T=Ta, N=N), wave=cab,
                      name=name or f'{a.name} \\circledast {b.name}')


def convolve(a, b, name=None):
    '''
    Convolve in frequency domain
    '''
    Ta = a.sampling.T
    Tb = b.sampling.T
    assert (Ta == Tb)

    Na = a.sampling.N
    Nb = b.sampling.N
    N = Na + Nb

    wa = numpy.zeros(N, dtype=a.wave.dtype)
    wb = numpy.zeros(N, dtype=b.wave.dtype)
    wa[:Na] = a.wave
    wb[:Nb] = b.wave
    sa = numpy.fft.fft(wa)
    sb = numpy.fft.fft(wb)
    cab = numpy.real(numpy.fft.ifft(sa*sb))
    return lmn.Signal(lmn.Sampling(T=Ta, N=N), wave=cab,
                      name=name or f'{a.name} \\circledast {b.name}')


Cs = [convolve(F, E) for F, E in zip(Fs, Es)]
Ts = [tconvolve(F, E) for F, E in zip(Fs, Es)]
Ps = [multiply_period(C) for C in Cs]

with pages("convo.pdf") as out:

    def page(sigs, name, **k):
        fig, (tx, fx) = plot_signals(sigs, **k)
        fx.set_yscale('log')
        plt.suptitle(f'${name}$')
        out.savefig()
        plt.clf()

    k = dict(drawstyle='steps-mid',
             linewidth='progressive',
             iunits='1',
             flim=(0, 5*units.MHz))

    page(Fs, "F-signals", **k)
    page(Es, "E-signals", **k)
    page(Cs, "Convolution (frequency)", **k)
    page(Ts, "Convolution (interval)", **k)
    page(Ps, "convolution \\cdot period", **k)
