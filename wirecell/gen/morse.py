#!/usr/bin/env python
'''Support for the "morse" pattern of depos.

This pattern constsis of a 2D rectangular but slightly irregular grid of depos
for each wire plane.  Each plane grid is offset in time from its neighbors by an
amount T.  T is chosen large enough so that the field and electronics response
(if not the longer RC response) does not lead to the ADC waveforms of two plane
grids of depos overlapping.

One axis of a plane grid of depos is parallel to the wires of its plane and the
other is orthogonal, along the pitch direction.  A set of closely spaced depos
run along the wire direction to form a short track.  An progressively-spaced
array of such tracks are then replicated in the pitch direction.  The location
in pitch of each subsequent track is N*pitch plus a growing n*impact from the
previous track.  N is chosen large enough so that field respone extent of two
neighboring tracks do not overlap.

Defining with nominal values:

- pitch jump P (25 pitches)
- impact jump I (0.1 fraction of a pitch)

We would produce near a wire "|" some depos ":" like:

    Target plane 0, at time = 0.5T

    0          P+i        2P+2I       3P+3I       4P+4I       5P+5I
    |          |          |          |          |          |
    |          |          |          |          |          |
    :          |:         | :        |  :       |   :      |    :
    :          |:         | :        |  :       |   :      |    :
    |          |          |          |          |          |
    |          |          |          |          |          |

    Target plane 1, at time = 1.5T

    ... ASCII picture omitted in the name of preserving sanity ...

    Target plane 2, at time = 2.5T

    ... ASCII picture omitted in the name of preserving sanity ...

'''
import json
import numpy
from pathlib import Path
from math import sqrt, pi
from wirecell import units
import dataclasses
from typing import List
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths
from wirecell.util.wires.array import mean_wire_pitch

def generate(plane_wires, refx, charge, length, planes=(0,1,2),
             time_jump=500*units.us, pitch_jump=25, impact_jump=0.1):


    nimpacts = 1 + int(round(0.5/impact_jump)) 


    datas = list()
    infos = list()

    ntot = 0
    # pwires is Nx2
    for plane in planes:
        pwires = plane_wires[plane]

        wvec, pvec = mean_wire_pitch(pwires)
        wdir = wvec / numpy.linalg.norm(wvec)
        pmag = numpy.linalg.norm(pvec)
        pdir = pvec/pmag

        nwires = pwires.shape[0]
        mid = pwires[nwires//2]

        origin = 0.5*(mid[0]+mid[1])
        origin[0] = refx

        t = time_jump * (0.5 + plane)

        for impnum in range(nimpacts):
            pdist = impnum*pmag*(pitch_jump + impact_jump)
            cen = origin + pdist*pdir
            half = 0.5 * wdir * length


            points = numpy.linspace(cen - half, cen+half, endpoint=True)
            npts = points.shape[0]

            charges = numpy.expand_dims(numpy.array((charge,)*npts), 1)
            sigmas = numpy.zeros(2*npts).reshape((-1,2))
            times = numpy.expand_dims(t + numpy.zeros(npts), 1)

            ids = numpy.expand_dims(numpy.arange(ntot, ntot+npts), 1)
            rest = numpy.array((0,0,0)*npts).reshape((-1,3))

            data = numpy.hstack((times,charges,points,sigmas), dtype='float32')
            info = numpy.hstack((ids,rest), dtype='int32')
            
            datas.append(data)
            infos.append(info)
            ntot += npts
            #print(f'{plane=} {impnum=} {npts=} {t=} {cen=}')
    data = numpy.vstack(datas)
    info = numpy.vstack(infos)
    return data, info


sqrt2pi = sqrt(2*pi)

def gauss(x, A, mu, sigma, *p):
    '''
    Gaussian distribution model for fitting.
    '''
    return A*numpy.exp(-0.5*((x-mu)/sigma)**2)/(sigma*sqrt2pi)


def load_depos(fname):
    fp = numpy.load(fname)
    d = fp["depo_data_0"]
    i = fp["depo_info_0"]
    ind = i[:,2] == 0
    return d[ind,:]

def load_frame(fname):
    fp = numpy.load(fname)
    # for now, assume frame is pret-a-porter
    return fp["frame_*_0"]

@dataclasses.dataclass
class WavePeak:
    '''
    Information about a peak in a 1D waveform.
    '''

    peak: int
    '''The where along the waveform the peak resides.'''

    fwhm: float
    '''Full-width of the peak at half-max.'''

    hh: float
    '''Half of the height of the peak.'''

    left: float
    '''The left side of the width measured in fractional indices.'''

    right: float
    '''The right side of the width measured in fractional indices.'''

    tot: float
    '''The sum of waveform values over the peak.'''

    mask: slice
    '''A mask that captures the peak.'''
    
    A: float
    '''The fit Gaussian normalization fit parameter.  See gauss().'''

    mu: float
    '''The fit Gaussian mean fit parameter.  See gauss().'''


    sigma: float
    '''The fit Gaussian sigma fit parameter.  See gauss().'''

    cov: numpy.ndarray
    '''The covariance matrix of the fit.'''

    @classmethod
    def from_dict(cls, data):
        fields = [f.name for f in dataclasses.fields(cls)]
        fields.remove("mask")
        fields.remove("cov")
        dat = {f:data.get(f) for f in fields}
        dat['mask'] = slice(*data.get("mask"))
        dat['cov'] = numpy.array(data.get("cov"))
        return cls(**dat)
            

def wave_peaks(wave, which_peaks=None, threshold=0.1):
    '''Return measures of peaks in wave.

    - which_peaks :: None considers all found, integer considers the peak at
      that index, list of integer select peaks.  If which_peaks is a scalar
      index, return the measure instead of list of measures.

    - threshold :: the minimum value relative the max sample value of wave that
      a sample must have to be considered at a peak.

    '''

    # Find the peaks
    peaks = find_peaks(wave, height = threshold*numpy.max(wave))[0]

    scalar = False
    if isinstance(which_peaks, int):
        which_peaks = [which_peaks]
        scalar = True
    if which_peaks is not None:
        peaks = [peaks[ind] for ind in which_peaks]

    # Characterize that peak full width half max 
    info = [numpy.array(peaks)]
    info += peak_widths(wave, peaks, rel_height=0.5)

    # the "x" values in fits below are simply indices.
    iota = numpy.arange(wave.size, dtype=int)

    # fit each peak
    ret = list()
    for peak, fwhm, hh, left, right in zip(*info):

        # Zero out activity outside of current peak.  Fixme: this assumes peaks
        # are well separated.
        mask = slice(int(round(left-fwhm)), int(round(right+fwhm)))
        tofit = numpy.zeros_like(wave)
        tofit[mask] = wave[mask]

        # Guess initial parameters
        A = numpy.sum(tofit)
        p0 = (A, peak, 0.5*fwhm)
        try:
            fit,cov = curve_fit(gauss, iota, tofit, p0=p0)
        except RuntimeError:
            nnz = numpy.sum(tofit>0)
            print(f'warning: fit failed with peak {peak}, {nnz} nonzero, {p0=}')
            fit=[None,None,None]
            cov=None


        one = WavePeak(peak=peak,fwhm=fwhm,hh=hh,left=left,right=right,
                       tot=A, mask=mask,
                       A = fit[0], mu = fit[1], sigma = fit[2],
                       cov = cov)

        ret.append(one)

    if scalar:
        return ret[0]
    return ret;


@dataclasses.dataclass
class FramePeaks:

    total: WavePeak
    '''The peak found after summing all channels in a plane.'''

    tick: List[WavePeak]
    '''The individual peaks in the tick direction for each impact position.'''

    chan: List[WavePeak]
    '''The indivdiual peaks in the channel direction for each impact position.'''

    @classmethod
    def from_dict(cls, data):
        dat=dict()
        dat['total'] = WavePeak.from_dict(data.get('total'))
        dat['tick'] = [WavePeak.from_dict(d) for d in data.get('tick')]
        dat['chan'] = [WavePeak.from_dict(d) for d in data.get('chan')]
        return cls(**dat)

def load_frame_peaks(src):
    import io
    if isinstance(src, str) and Path(src).exists():    # filename
        src = open(src)
    if isinstance(src, io.IOBase): # file object
        src = src.read()
    if isinstance(src, str):    # file contents
        src = json.loads(src)
    if isinstance(src, dict):   # a single frame peak
        return FramePeaks.from_dict(src)
    if isinstance(src, list):   # list of frame peaks
        return [FramePeaks.from_dict(d) for d in src]
    raise ValueError(f'unsupported type: {type(src)}')
    

class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        if isinstance(obj, numpy.floating):
            return float(obj)
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()        
        if isinstance(obj, slice):
            return (obj.start, obj.stop, obj.step)

        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return super().default(obj)

def dump_frame_peaks(dst, peaks):
    if isinstance(dst, str): # filename
        dst = open(dst, "w")
    dst.write(json.dumps(peaks, indent=4, cls=Encoder))

def scale_slice(s, r):
    d = s.stop - s.start
    return slice(int(s.start - r*d), int(s.stop + r*d))
    

def patch_chan_mask(fp):
    '''
    Return a mask that spans the .chan masks.
    '''
    mask = None
    for c in fp.chan:
        if mask is None:
            mask = [c.mask.start, c.mask.stop]
            continue
        if mask[0] > c.mask.start:
            mask[0] = c.mask.start
        if mask[1] < c.mask.stop:
            mask[1] = c.mask.stop
    return slice(mask[0], mask[1])




def frame_peaks(signal, channel_ranges):
    '''Return list of FramePeaks for peaks found in a frame with a "morse"
    pattern of activity.

    Each FramePeaks has attributes holding one or a list of WavePeaks:

    - .total :: the WavePeak in tick for total of targeted activity in plane.
    
    - .tick :: list of WavePeak in tick, each for track at one impact position

    - .chan :: list of WavePeak in chan, each for track at one impact position

    '''
    pms = list()
    for plane, (ch1,ch2) in enumerate(zip(channel_ranges[:-1],channel_ranges[1:])):

        # the channels of this plane
        chan_mask = slice(ch1,ch2)

        # Total plane activity waveform
        tot_wave = numpy.sum(signal[chan_mask,:], axis=0)

        # Total tick peak measure selected as the plane'th peak which holds
        # activity that targeted this plane.
        ttpm = wave_peaks(tot_wave, plane)

        # Next we measure individual activities across the channel dimension.

        # Narrow to just the target activity for this plane
        patch = numpy.zeros_like(signal)
        patch[chan_mask, ttpm.mask] = signal[chan_mask, ttpm.mask]
        
        # Form a "channel waveform" along the peak tick.
        chan_wave = patch[:,ttpm.peak]

        # Find the peaks across channels at the peak tick.  Nominally should be
        # 6, one for each impact position with default "--impact-jump" option of
        # "depo-morse".
        cpms = wave_peaks(chan_wave)

        # Finally, use each channel peak to select a per impact position waveform.
        itpms = [wave_peaks(patch[cpm.peak,:], 0) for cpm in cpms]

        fp = FramePeaks(total=ttpm, tick=itpms, chan=cpms)
        pms.append(fp)

    return pms

