#!/usr/bin/env python
'''
Define a data schema for noise related information.

See wire-cell-toolkit/aux/docs/noise.org for more information.

'''

from collections import namedtuple

class NoiseSpectrum_v2(namedtuple("NoiseSpectrum", "period nsamples gain shaping plane wirelen const freqs amps group")):
    '''
    This new class has not been used in any other python script so far. 
    Information about mean spectral amplitude for noise models.

    # optional
    :param float gain: the gain in units of [voltage]/[charge] assumed in the spectrum
    :param float shaping: the shaping time of the amplifier in units of [time] assumed in the spectrum
    :param int plane: the identifier for the plane holding the wire from which this noise is associated.
    :param float wirelen: the length of wire in units of [length] for the wire assumed to produce the noise.
    :param float const: mean spectral amplitude of a white noise component
    :param int group: an identifier of a group of channels to which this spectrum applies
    # required
    :param float period: the sampling period in units of [time] used to make the spectrum
    :param int nsamples: the number of time sampled used to make the spectrum
    :param list freqs: list of floating point sampled frequency in units of [frequency] (not MHZ!)
    :param list amps: list of floating point sampled noise amplitude in units of [voltage]/[frequency].
    '''
    __slots__ = ()


    
class NoiseSpectrum(namedtuple("NoiseSpectrum", "period nsamples gain shaping plane wirelen const freqs amps")):
    '''
    This is for MicroBooNE and any other experiment incoherent noise simulation with no argument "group". 
    ICARUS incoherent noise simulation also uses this class "NoiseSpectrum" other than "NoiseSpectrum_v2".

    Information about mean spectral amplitude for noise models.

    # optional
    :param float gain: the gain in units of [voltage]/[charge] assumed in the spectrum
    :param float shaping: the shaping time of the amplifier in units of [time] assumed in the spectrum
    :param int plane: the identifier for the plane holding the wire from which this noise is associated.
    :param float wirelen: the length of wire in units of [length] for the wire assumed to produce the noise.
    :param float const: mean spectral amplitude of a white noise component
    # required
    :param float period: the sampling period in units of [time] used to make the spectrum
    :param int nsamples: the number of time sampled used to make the spectrum
    :param list freqs: list of floating point sampled frequency in units of [frequency] (not MHZ!)
    :param list amps: list of floating point sampled noise amplitude in units of [voltage]/[frequency].
    '''
    __slots__ = ()
