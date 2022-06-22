#!/usr/bin/env python3
'''
Some simple DFT tests
'''

import numpy
import matplotlib
import matplotlib.pyplot as plt
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 8}

matplotlib.rc('font', **font)
nbins = 100
# interval space impulse function
impulse_inter = numpy.zeros(nbins, dtype='complex')
# frequency space spikes, a DC level
dc_freq = numpy.zeros(nbins, dtype='complex')
# DC plus higest frequency bin, ignoring any symmetry due to real-valued interval-spaceness
hilo_freq = numpy.zeros(nbins, dtype='complex')
lolo_freq = numpy.zeros(nbins, dtype='complex')

impulse_inter[10] = 1
dc_freq[0] = 1
hilo_freq[0] = 1
hilo_freq[-1] = 1
lolo_freq[0] = 1
lolo_freq[1] = 1

impulse_freq = numpy.fft.fft(impulse_inter)
dc_inter = numpy.fft.ifft(dc_freq)
hilo_inter = numpy.fft.ifft(hilo_freq)
lolo_inter = numpy.fft.ifft(lolo_freq)

# array type x (real(inter), amp(freq), phase(freq), real(freq), imag(freq))

def do_one(ax, c, title):
    ax[0].set_ylabel(title)

    ax[0].set_title("amplitude")
    ax[0].plot(numpy.abs(c))

    ax[1].set_title("phase")
    ax[1].plot(numpy.angle(c))

    ax[2].set_title("real")
    ax[2].plot(numpy.real(c))

    ax[3].set_title("imag")
    ax[3].plot(numpy.imag(c))


fig, axes = plt.subplots(8, 4)
do_one(axes[0], impulse_inter, "impulse inter")
do_one(axes[1], impulse_freq,  "impulse freq")
do_one(axes[2], dc_inter,      "dc inter")
do_one(axes[3], dc_freq,       "dc freq")
do_one(axes[4], hilo_inter,    "hilo inter")
do_one(axes[5], hilo_freq,     "hilo freq")
do_one(axes[6], lolo_inter,    "lolo inter")
do_one(axes[7], lolo_freq,     "lolo freq")

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(8.5,11)
plt.tight_layout()
plt.savefig("test_dft.pdf", dpi=100)
