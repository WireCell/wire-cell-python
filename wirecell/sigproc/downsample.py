#!/usr/bin/env python
'''
downsampling
'''
import response
from .. import units

import numpy
#import scipy.signal
import matplotlib.pyplot as plt


electron_charge = -1.6021766199999996e-19 # coulomb
def checkit(field_rf, charge = electron_charge):

    gain=14*1e-3/1e-15     # 14e12 Volt/Coulomb
    shaping=2*units.us

    t0, t1 = field_rf.times[0:2]
    dt = t1-t0
    tf = dt*field_rf.nbins    

    print "dt=%f us, time range: %f (%f us)" % (dt/units.us, tf, tf/units.us)

    nbins_hirez = 5000
    nbins_native = field_rf.nbins                      # 1000
    nbins_tick = int(dt*nbins_native / (0.5*units.us)) # 200
    n_downsample = nbins_hirez//nbins_tick             # 25

    print 'nbins:', nbins_tick, nbins_native, nbins_hirez

    tbin_hirez = tf/nbins_hirez
    tbin_native = tf/nbins_native
    tbin_tick = tf/nbins_tick
    
    print 'tbins (us):', tbin_tick/units.us, tbin_native/units.us, tbin_hirez/units.us

    times_hirez  = numpy.linspace(t0,tf,nbins_hirez)
    times_native = numpy.linspace(t0,tf,nbins_native)
    times_tick   = numpy.linspace(t0,tf,nbins_tick)

    elect_hirez = response.electronics(times_hirez, gain, shaping)
    elect_tick = response.electronics(times_tick, gain, shaping)

    field_hirez_rf = field_rf.resample(nbins_hirez)
    field_tick_rf = field_rf.resample(nbins_tick)

    charge_native = field_rf.response*(tbin_native/units.s)
    charge_hirez  = field_hirez_rf.response*(tbin_hirez/units.s)
    charge_tick   = field_tick_rf.response*(tbin_tick/units.s)

    charge_native = charge_native*charge/numpy.sum(charge_native)
    charge_hirez  = charge_hirez*charge/numpy.sum(charge_hirez)
    charge_tick   = charge_tick*charge/numpy.sum(charge_tick)

    charge_tots = map(numpy.sum, [charge_native, charge_hirez, charge_tick])
    print 'total charge', charge_tots

    # upsampled FFT
    elect_hirez_spec = numpy.fft.fft(elect_hirez)
    charge_hirez_spec = numpy.fft.fft(charge_hirez)

    # upsample, fft, mult, ifft, downsample
    convo_hirez = numpy.fft.ifft(elect_hirez_spec * charge_hirez_spec)
    convo_downsample = convo_hirez[::n_downsample]

    # upsample, fft, truncate, mult, ifft
    nbins_tick_half = nbins_tick//2
    elect_trunc_spec = numpy.hstack((elect_hirez_spec[:nbins_tick_half], elect_hirez_spec[-nbins_tick_half:]))
    charge_trunc_spec = numpy.hstack((charge_hirez_spec[:nbins_tick_half], charge_hirez_spec[-nbins_tick_half:]))
    convo_trunc = numpy.fft.ifft(elect_trunc_spec * charge_trunc_spec)
    convo_trunc /= n_downsample

    #convo_numpy = numpy.convolve(elect_hirez, charge_hirez, "same")
    #convo_scipy = scipy.signal.convolve(elect_hirez, charge_hirez, "same")


    charge_delta = numpy.zeros_like(elect_hirez)
    charge_delta[100] = 1.0

    print len(charge_delta), numpy.sum(numpy.abs(charge_delta))
    charge_delta_spec = numpy.fft.fft(charge_delta)
    convo_delta = numpy.fft.ifft(charge_delta_spec * elect_hirez_spec)
    convo_delta2 = numpy.convolve(charge_delta, elect_hirez)

    fig, axes = plt.subplots(3,4)
    time_range = (70,100)

    ax = axes[0,0]
    ax.plot(times_native/units.us, field_rf.response)
    ax.set_autoscalex_on(False)
    ax.set_xlim(time_range)
    ax.set_title('field resp')
    #ax.set_ylabel('Current')


    ax = axes[0,1]
    ax.plot(times_native/units.us, charge_native)
    ax.set_autoscalex_on(False)
    ax.set_xlim(time_range)
    ax.set_title('charge/.1us')
    #ax.set_ylabel('Current')

    ax = axes[0,2]
    ax.plot(times_hirez/units.us, charge_hirez)
    ax.set_autoscalex_on(False)
    ax.set_xlim(time_range)
    ax.set_title('charge/.02us')
    #ax.set_ylabel('Current')

    ax = axes[0,3]
    ax.plot(times_hirez/units.us, elect_hirez)
    ax.set_autoscalex_on(False)
    ax.set_xlim((0,10))
    ax.set_title('elect respon')
    #ax.set_ylabel('mV/fC')

    ax = axes[1,0]
    ax.semilogy(numpy.absolute(charge_hirez_spec[:nbins_tick]))
    ax.set_title("mag(fft(field))")

    ax = axes[1,1]
    ax.plot(numpy.angle(charge_hirez_spec[:nbins_tick]))
    ax.set_title("phase(fft(field))")

    ax = axes[1,2]
    ax.semilogy(numpy.absolute(elect_hirez_spec[:nbins_tick]))
    ax.set_title("mag(fft(elect))")

    ax = axes[1,3]
    ax.plot(numpy.angle(elect_hirez_spec[:nbins_tick]))
    ax.set_title("phase(fft(elect))")


    ax = axes[2,0]
    ax.plot(times_hirez/units.us, convo_hirez/1e-6)
    ax.set_xlim(time_range)
    ax.set_title('convo hirez [uV]')
    ax.set_ylabel('')

    ax = axes[2,1]
    ax.plot(times_tick/units.us, convo_downsample/1e-6)
    ax.plot(times_tick/units.us, convo_trunc/1e-6)
    ax.set_xlim(time_range)
    ax.set_title('convo ds&trunc [uV]')

    #ax = axes[2,2]
    #ax.plot(times_hirez/units.us, convo_numpy)
    #ax.plot(times_hirez/units.us, convo_scipy)
    #ax.set_xlim(time_range)
    #ax.set_title('convo (numpy)')
            

    ax = axes[2,2]
    ax.plot(times_hirez/units.us, charge_delta)
    ax.set_xlim((0,10))

    ax = axes[2,3]
    ax.plot(times_hirez/units.us, convo_delta)
    ax.plot(times_hirez/units.us, convo_delta2[:5000])
    ax.set_xlim((0,10))
