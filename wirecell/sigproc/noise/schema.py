#!/usr/bin/env python
'''
Define a data schema for noise related information.
'''

from collections import namedtuple

class NoiseSpectrum(namedtuple("NoiseSpectrum", "period nsamples gain shaping plane wirelen const freqs amps")):
    '''
    Information about average noise.

    :param float period: the sampling period in units of [time] used to make the spectrum
    :param int nsamples: the number of time sampled used to make the spectrum
    :param float gain: the gain in units of [voltage]/[charge] assumed in the spectrum
    :param float shaping: the shaping time of the amplifier in units of [time] assumed in the spectrum
    :param int plane: the identifier for the plane holding the wire from which this noise is associated.
    :param float wirelen: the length of wire in units of [length] for the wire assumed to produce the noise.
    :param list freqs: list of floating point sampled frequency in units of [frequency] (not MHZ!)
    :param list amps: list of floating point sampled noise amplitude in units of [voltage]/[frequency].
    '''
    __slots__ = ()


    
