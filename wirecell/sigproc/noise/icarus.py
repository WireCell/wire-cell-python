#!/usr/bin/env python
'''
Vanilla noise loader for ICARUS. It will probably require some adjustmens when
more realistic noise details are available.
'''

from wirecell import units, util
from . import schema

import numpy

def load_noise_spectra(filename):
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
    print(meta[0])
    period = 1.0/util.unitify(meta[0], 'megahertz')
    print("Period %d" % period)
    nsamples = int(meta[2])
    print("nsamples %d" % nsamples)
    gain = util.unitify(meta[4], meta[5])
    print("gain %.4f" % gain)
    shaping = util.unitify(meta[6], meta[7])
    print("shaping %.4f" % shaping)
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

    print(len(freq))
    print(len(amps))

    # TODO: NoiseSpectrum schema must contain also anode sorting for ICARUS...
    for iwire in range(nwires):
        ns = schema.NoiseSpectrum(period, nsamples, gain, shaping,
                                      planes[iwire], wirelens[iwire],
                                      consts[iwire], list(freq), list(amps[iwire]))
        noises.append(ns)

    noises.sort(key = lambda s: 100*(s.plane+1) + s.wirelen/units.meter)
    return noises


def load_coherent_noise_spectra(filename):
    '''
    Load a noise spectra file in format

    <sampfreq> <frequnits> <number> Ticks <gain> <gainunits> <shaping> <timeunit>
    <Group> < wire-delta1 >  <  wire-delta2  > ...
    -1      <constantterm1>  <  constantterm2> ...
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
    print(meta[0])
    period = 1.0/util.unitify(meta[0], 'megahertz')
    print("Period %d" % period)
    nsamples = int(meta[2])
    print("nsamples %d" % nsamples)
    gain = util.unitify(meta[4], meta[5])
    print("gain %.4f" % gain)
    shaping = util.unitify(meta[6], meta[7])
    print("shaping %.4f" % shaping)

    groups = [float(v) for v in lines[1].split()[1:]]
    consts = [float(v)*units.mV for v in lines[2].split()[1:]]
    data = list()
    for line in lines[3:]:
        data.append([float(v) for v in line.split()])
    data = numpy.asarray(data)
    freq = data[:,0]*util.unitify("1.0", meta[1])
    amps = data[:,1:].T*units.mV
    ngroups = len(groups)
    noises = list ()

    for igroup in range(ngroups):
        ns = { 'period' : period,
          'nsamples' : nsamples,
          'gain' : gain,
          'shaping' : shaping,
          'wire-delta': groups[igroup],
          'const' : consts[igroup],
          'freqs' : list(freq),
           'amps' : list(amps[igroup])
         }
        noises.append(ns)

    #print(noises)
    return noises
