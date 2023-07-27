#!/usr/bin/env pytest
'''
Test wirecell.sigproc.fwd module
'''
from pytest import approx
from math import isclose
import numpy
from numpy.fft import ifft
from wirecell import units
from wirecell.sigproc import fwd

def test_signal():
    times = fwd.make_times(100, 1)
    assert len(times) == 100
    sig = fwd.Signal("test", times)
    assert len(sig) == 100
    assert sum(sig.waveform) == 0
    sig += 1
    assert sum(sig.waveform) == 100
    assert len(sig) == 100
    sig *= 1
    assert sum(sig.waveform) == 100
    assert len(sig) == 100
    a = sig + sig
    assert sum(a.waveform) == 200
    assert len(a) == 100

def test_rebin():
    times = fwd.make_times(100, 1)
    sig = fwd.gauss_signal(times, sigma=3, tpeak=times[50])
    sig2 = fwd.rebin(sig, 5)
    assert(approx(sig.wave_energy, 1e-4) == sig2.wave_energy)

def test_convolution():
    times = fwd.make_times(100, 1)
    values = numpy.zeros(100)
    values[25] = 1;
    sig1 = fwd.Signal("sig1", times, values)
    assert numpy.sum(sig1.waveform) == 1
    sig2 = fwd.Signal("sig2", times, fwd.gauss_wave(times, sigma=3, tpeak=times[25]))
    assert approx(numpy.sum(sig2.waveform)) == 1
    cwave = numpy.convolve(sig1.waveform, sig2.waveform, "full")
    assert approx(numpy.sum(cwave)) == 1    
    convo = fwd.Convolution("convo", sig1, sig2)
    assert convo.waveform.size == 100
    assert approx(numpy.sum(convo.waveform)) == 1
    

def test_gauss():
    N=1024
    T=0.5*units.us
    times = fwd.make_times(N, T)
    sigma = numpy.array((1,2,4,8))*units.us
    g = fwd.gauss_wave(times, sigma=sigma, tpeak=times[N//2])
    assert all([approx(x) == 1 for x in numpy.sum(g, axis=1)])
    
    sigma = 2*units.us
    g = fwd.gauss_wave(times, sigma=sigma, tpeak=times[N//2])
    assert approx(numpy.sum(g)) == 1

    
def test_electronics_response():
    for det in fwd.known_responses:
        times = numpy.arange(0, 20*units.us, 0.1*units.us)
        er = fwd.ElecResponse("elec", times, det)
        for one in ('elec_type', 'peak_gain', 'shaping'):
            assert getattr(er, one) == fwd.known_responses[det][one]

        wave = er.waveform
        assert wave.shape == times.shape
        assert numpy.sum(wave[-10:]) == 0

        
def test_field_response():
    for det in fwd.known_responses:
        fr = fwd.FieldResponse('field', det)
        assert(fr)
        # why is period not closer to canonical?
        eps = 2e-5
        assert isclose(fr.T, 0.1*units.us, rel_tol=eps)
        assert isclose(fr.sampling_hz, 10e6, rel_tol=eps)

def test_detector_response():
    for det in fwd.known_responses:
        dr = fwd.DetectorResponse('det', det)
        eps = 1e-6

        # we iff()' the spec just to take care of FFT normalization
        assert isclose(dr.wave_energy, dr.spec_energy, rel_tol=eps)
        assert isclose(dr.er.wave_energy, dr.er.spec_energy, rel_tol=eps)
        assert isclose(dr.fr.wave_energy, dr.fr.spec_energy, rel_tol=eps)

def test_noise():
    N = 1024
    T = units.us
    times = fwd.make_times(N, T)
    assert len(times) == N

    for det in fwd.known_responses:
        noise = fwd.Noise("noise", times, det)
        assert len(noise.spectrum) == N
        assert len(noise.amplitude) == N
        w1 = noise.waveform
        assert isinstance(w1, numpy.ndarray)
        assert len(w1) == N
        w2 = noise.waveform
        assert len(w1) == N
        assert numpy.sum(w1) != numpy.sum(w2)
