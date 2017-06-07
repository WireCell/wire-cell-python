#!/usr/bin/env python
'''
Stuff to deal with MicroBooNE specific noise things and such.
'''

from wirecell import units, util
import schema

import numpy


def load_noise_spectra_v1(filename):
    '''
    Load a noise spectra file in format
    
    <sampfreq> <frequnits> <number> Ticks <gain> <gainunits> <shaping> <timeunit>
    Freq  <wirelenghtincm1>  <wirelenghtincm2> ...
    Plane <       planeid1>  <       planeid2> ...
    -1    <  constantterm1>  <  constantterm1> ...
    <freq0> <amplitude1[0]>  <  amplitude2[0]> ...
    <freq1> <amplitude1[1]>  <  amplitude2[1]> ...
    <freqN> <amplitude1[N]>  <  amplitude2[N]> ...

    Also ignore lines starting with '#' or empty
    '''
    lines=list()
    with open(filename) as fp:
        for line in fp.readlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            lines.append(line)
    meta = lines[0].split()
    period = 1.0/util.unitify(meta[0], meta[1])
    nsamples = int(meta[2])
    gain = util.unitify(meta[4], meta[5])
    shaping = util.unitify(meta[6], meta[7])
    wirelens = [float(v)*units.cm for v in lines[1].split()[1:]]
    planes = [int(v) for v in lines[2].split()[1:]]
    consts = [float(v)*units.mV for v in lines[3].split()[1:]]
    data = list()
    for line in lines[4:]:
        data.append([float(v) for v in line.split()])
    data = numpy.asarray(data)
    freq = data[:,0]*util.unitify("1.0", meta[1])
    amps = data[:,1:].T*units.mV
    nwires = len(wirelens)
    noises = list ()
    for iwire in range(nwires):
        ns = schema.NoiseSpectrum(period, nsamples, gain, shaping,
                                      planes[iwire], wirelens[iwire],
                                      consts[iwire], list(freq), list(amps[iwire]))
        noises.append(ns)
        if planes[iwire] == 1:            # v1 implicitly equates U and V planes but only provides plane=1 data
            ns0 = schema.NoiseSpectrum(period, nsamples, gain, shaping,
                                           0, wirelens[iwire],
                                           consts[iwire], list(freq), list(amps[iwire]))

            noises.append(ns0)
    noises.sort(key = lambda s: 100*(s.plane+1) + s.wirelen/units.meter)
    return noises
