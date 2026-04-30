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
    Some identifier, often the "frame_tag_event" name inside a frame file archive.
    '''
    
    tag: str = ""
    '''
    A marker that gives a frame some semantic interpretation.
    '''

    index: int = 0
    '''
    The identify this frame in some larger set or sequence.
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

    @property
    def summary(self):
        return dict(tag=self.tag, index=int(self.index),
                    nchans = int(self.nchans), nticks = int(self.nticks),
                    chidmin = int(self.chan_bounds[0]), 
                    chidmax = int(self.chan_bounds[1]), 
                    tref=self.tref, duration=self.duration,
                    period=self.period, tbin=int(self.tbin))


def wan_pad(frame, detector, detname=None):
    '''Partition and zero-pad a frame into per-anode frames in WAN order.

    For each anode in *detector* that owns at least one channel present in
    *frame*, a new :class:`Frame` is returned whose rows span **all** channels
    of that anode in WAN order (face → plane → pitch).  Rows for channels not
    present in *frame* are filled with zeros.

    Parameters
    ----------
    frame : Frame
        Source frame.  ``frame.channels`` may be any subset of the channels
        that the detector knows about.
    detector : dict
        Full detector object as returned by ``wirecell.util.wires.info.todict()``
        (one element of that list), i.e. ``{"ident": …, "anodes": […]}``.
    detname : str or None
        Detector identifier forwarded to ``wan.anode_faces()``; reserved for
        future face-ordering changes, currently unused.

    Returns
    -------
    dict
        ``{anode_ident (int): Frame}`` — one entry per anode that contributed
        channels.  Each Frame shares ``period``, ``tref``, ``tbin``, and
        ``name`` with the input frame.
    '''
    from wirecell.util.wires import wan

    # Identify which frame channels belong to which anode.
    partition = wan.anode_partition(detector, frame.channels)

    anode_by_ident = {a['ident']: a for a in detector['anodes']}
    src_row = frame.chids2row        # {chid: row index in frame.samples}

    result = {}
    for anode_ident, present_chids in partition.items():
        a = anode_by_ident[anode_ident]
        ordered = wan.anode_chids(a, det=detname)

        out_samples = numpy.zeros((len(ordered), frame.nticks),
                                  dtype=frame.samples.dtype)
        for out_row, chid in enumerate(ordered):
            if chid in src_row:
                out_samples[out_row] = frame.samples[src_row[chid]]

        result[anode_ident] = Frame(
            samples=out_samples,
            channels=numpy.array(ordered, dtype=frame.channels.dtype),
            period=frame.period,
            tref=frame.tref,
            tbin=frame.tbin,
            name=frame.name,
        )
    return result


@dataclasses.dataclass
class FrameArrayIdentifier:

    kind : str 
    '''
    What kind of array: frame, channels or tickinfo
    '''

    tag: str = ""
    '''
    The tag string, may contain underscores
    '''

    index: int = 0
    '''
    A count, eg an "event number", for a sequence of frames of common kind and tag.
    '''

def frame_array_identifier(fname):
    '''
    Return a FrameArrayIdentifier for a given file name
    '''
    base = fname.rsplit(".", 1)
    parts = base[0].split("_")
    return FrameArrayIdentifier(
        kind = parts[-1],
        tag = '_'.join(parts[1:-1]), # tags may have inner underscores
        index = int(parts[-1]))
    

def load(fp):
    '''
    Yield frame objects in a "frame file" given by fp.

    fp is file name as string or pathlib.Path or ario/numpy.load() like.
    '''
    if isinstance(fp, str):
        fp = ario.load(fp)

    frame_names = [key for key in fp if key.startswith("frame_")]
    for frame_name in frame_names:
        fai = frame_array_identifier(frame_name)

        ti = fp[f'tickinfo_{fai.tag}_{fai.index}']
        ch = fp[f'channels_{fai.tag}_{fai.index}']

        yield Frame(samples=fp[frame_name], channels=ch,
                    period=ti[1], tref=ti[0], tbin=int(ti[2]),
                    name=frame_name, tag=fai.tag, index=fai.index)
