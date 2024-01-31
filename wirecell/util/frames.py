#!/usr/bin/env python
'''
Some helpers for frame like objects
'''

from . import ario
import numpy
import dataclasses
from wirecell import units

@dataclasses.dataclass
class Frame:

    samples : numpy.ndarray | None = None
    '''
    Frame samples as 2D array shape (nchans, nticks).
    '''

    channels : numpy.ndarray | None = None
    '''
    Array of channel identity numbers.
    '''
    
    period: float | None = None
    '''
    The time-domain sampling period (aka "tick").
    '''

    tref: float = 0.0
    '''
    The reference time.
    '''
    
    tbin: int = 0
    '''
    The time bin represented by the first column of samples array.
    '''

    name: str = ""
    '''
    Some human identifier.
    '''
    
    @property
    def nchans(self):
        '''
        Number of channels
        '''
        return self.samples.shape[0]

    @property
    def chan_bounds(self):
        '''
        Pair of min/max channel number
        '''
        return (numpy.min(self.channels), numpy.max(self.channels))

    @property
    def nticks(self):
        '''
        Number of samples in time
        '''
        return self.samples.shape[1]

    @property
    def tstart(self):
        '''
        The time of the first sample
        '''
        return self.tref + self.period*self.tbin

    @property
    def duration(self):
        '''
        The time spanned by the samples
        '''
        return self.nticks*self.period

    @property
    def absolute_times(self):
        '''
        An array of absolute times of samples
        '''
        t0 = self.tstart
        tf = self.duration
        return numpy.linspace(t0, tf, self.nticks, endpoint=False)

    @property
    def times(self):
        '''
        An array of times of samples relative to first
        '''
        t0 = 0
        tf = self.duration
        return numpy.linspace(t0, tf, self.nticks, endpoint=False)

    @property
    def Fmax_hz(self):
        '''
        Max sampling frequency in hz
        '''
        T_s = self.period/units.s
        return 1/T_s

    @property
    def freqs_MHz(self):
        '''
        An array of frequencies in MHz of Fourier-domain samples
        '''
        return numpy.linspace(0, self.Fmax_hz/1e6, self.nticks, endpoint=False)

    @property
    def chids2row(self):
        '''
        Return a channel ID to row index
        '''
        return {c:i for i,c in enumerate(self.channels)}

    def waves(self, chans):
        '''
        Return waveform rows for channel ID or sequence of IDs in chans.
        '''
        scalar = False
        if isinstance(chans, int):
            chans = [chans]
            scalar = True

        lu = self.chids2row

        nchans = len(chans)
        shape = (nchans, self.nticks)
        ret = numpy.zeros(shape, dtype=self.samples.dtype)
        for ind,ch in enumerate(chans):
            row = lu[ch]
            ret[ind,:] = self.samples[row]
        if scalar:
            return ret[0]
        return ret

    def __str__(self):
        return f'({self.nchans}ch,{self.nticks}x{self.period/units.ns:.0f}ns) @ {self.tstart/units.us:.0f}us'

def load(fp):
    '''
    Yield frame objects in fp.

    fp is file name as string or pathlib.Path or ario/numpy.load() like.
    '''
    if isinstance(fp, str):
        fp = ario.load(fp)

    frame_names = [key for key in fp if key.startswith("frame_")]
    for frame_name in frame_names:
        _,tag,num = frame_name.split("_")

        ti = fp[f'tickinfo_{tag}_{num}']
        ch = fp[f'channels_{tag}_{num}']

        yield Frame(samples=fp[frame_name], channels=ch,
                    period=ti[1], tref=ti[0], tbin=int(ti[2]),
                    name=frame_name)
