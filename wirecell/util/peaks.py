#!/usr/bin/env python
'''
Find and represent peaks in 1D and 2D frame and waveform arrays.
'''
import numpy
import numpy.ma as ma
import json
from pathlib import Path
import dataclasses
from typing import List, Tuple
from math import sqrt, pi

try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = np.ndarray[int, np.dtype[float]]

from wirecell.util.codec import dataclass_dictify
from wirecell.util.bbox import union as union_bbox

import logging
log = logging.getLogger("wirecell.util")

sqrt2pi = sqrt(2*pi)

def gauss(x, A, mu, sigma, *p):
    '''
    Gaussian distribution model for fitting.
    '''
    return A*numpy.exp(-0.5*((x-mu)/sigma)**2)/(sigma*sqrt2pi)

@dataclasses.dataclass
class BaselineNoise:
    '''
    Characterize baseline (or any) noise.
    '''

    A : float
    '''
    Normalization (fit constant)
    '''

    mu : float
    '''
    Mean (fit mean)
    '''

    sigma : float
    '''
    Width (fit standard deviation)
    '''

    N : int
    '''
    Number of samples
    '''

    C : float
    '''
    Normalization (sum)
    '''

    avg : float
    '''
    Average
    '''

    rms : float
    '''
    Width (calculated)
    '''

    med : float
    '''
    Median
    '''

    lo : float
    '''
    34% quantile below the median
    '''

    hi : float
    '''
    34% quantile above the median
    '''

    cov: ArrayLike | None = dataclasses.field(default_factory=lambda: None)
    '''
    Covariance matrix of fit.  Non implies A,mu,sigma are statistical.
    '''

    hist: tuple | None = None
    '''
    The bin content and edges of the histgram that was fit
    '''

def baseline_noise(array, bins=200, vrange=100):
    '''Return a BaselineNoise derived from array a.

    This attempts to fit a Gaussian model to a histogram of array values
    spanning given number of bins of value range given by vrange.  The vrange
    defines an extent about the MEDIAN VALUE.  If it is a tuple it gives this
    extent explicitly or if scalar the extent is symmetric, ie median+/-vrange.

    This will raise exceptions:

    - ZeroDivisionError when the signal in the vrange is zero.

    - RuntimeError when the fit fails.

    '''
    from scipy.optimize import curve_fit

    nsig = len(array)
    lo, med, hi = numpy.quantile(array, [0.5-0.34,0.5,0.5+0.34])

    if not isinstance(vrange, tuple):
        vrange=(med-vrange, med+vrange)
    vrange=(med+vrange[0], med+vrange[1])

    hist = numpy.histogram(array, bins=bins, range=vrange)
    counts, edges = hist

    C = numpy.sum(counts)
    avg = numpy.average(edges[:-1], weights=counts)
    rms = sqrt(numpy.average((edges[:-1]-avg)**2, weights=counts))

    p0 = (A,mu,sig) = (C,avg,rms)

    try:
        (A,mu,sig),cov = curve_fit(gauss, edges[:-1], counts, p0=p0)
    except RuntimeError:
        cov = None

    return BaselineNoise(A=A, mu=mu, sigma=sig,
                         N=nsig,
                         C=C, avg=avg, rms=rms,
                         med=med, lo=lo, hi=hi,
                         cov=cov, hist=hist)

@dataclasses.dataclass
@dataclass_dictify
class Peak1d:
    '''
    Information about a peak in a 1D array.
    '''

    peak: int = 0
    '''The where along the waveform the peak resides.'''

    fwhm: float = 0.0
    '''Full-width of the peak at half-max.'''

    hh: float = 0.0
    '''Half of the height of the peak.'''

    left: float = 0.0
    '''The left side of the width measured in fractional indices.'''

    right: float = 0.0
    '''The right side of the width measured in fractional indices.'''

    tot: float = 0.0
    '''The sum of waveform values over the peak.'''

    mask: slice = dataclasses.field(default_factory=lambda: slice(0,0))
    '''A mask that captures the peak.'''
    
    A: float = 0.0
    '''The fit Gaussian normalization fit parameter.  See gauss().'''

    mu: float = 0.0
    '''The fit Gaussian mean fit parameter.  See gauss().'''

    sigma: float = 0.0
    '''The fit Gaussian sigma fit parameter.  See gauss().'''

    cov: ArrayLike | None = dataclasses.field(default_factory=lambda: None)
    '''The covariance matrix of the fit.'''


def find1d(wave, npeaks=None, threshold=0):
    '''Return measures of peaks in 1d waveform.

    - npeaks :: return only the number npeaks highest peaks.  None returns all.

    - threshold :: the minimum value for a sample to be considered a peak.

    '''
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks, peak_widths

    # Find the peaks
    peaks = find_peaks(wave, height = threshold)[0]
    peaks = list(sorted(peaks, key=lambda p: wave[p]))
    if npeaks is not None:
        peaks = peaks[0:npeaks]

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
            fit=p0
            cov=None

        one = Peak1d(peak=peak,fwhm=fwhm,hh=hh,left=left,right=right,
                     tot=A, mask=mask,
                     A = fit[0], mu = fit[1], sigma = fit[2],
                     cov = cov)

        ret.append(one)

    return ret;


@dataclasses.dataclass
class Plateaus:

    number: int = 0
    '''
    The number of objects.
    '''

    @property
    def labels(self):
        'Array of labels (1-based counts)'
        return 1 + self.indices
    @property
    def indices(self):
        'Array of indices (0-based counts)'
        return numpy.arange(self.number)

    labeled: ArrayLike = dataclasses.field(default_factory=lambda: numpy.zeros((0,0)))
    '''
    The frame with labeled pixels for each object.
    '''

    bboxes: List[Tuple[slice]] = ()
    '''
    Bounding boxes of each object.
    '''

    sums: ArrayLike = dataclasses.field(default_factory=lambda: numpy.zeros((0,)))
    '''
    The total value of each object.
    '''

    counts: ArrayLike = dataclasses.field(default_factory=lambda: numpy.zeros((0,)))
    '''
    The number of pixels of each object.
    '''

    coms: ArrayLike = dataclasses.field(default_factory=lambda: numpy.zeros((0,0)))
    '''
    The center of mass of objects in pixel space.
    '''

    threshold: float = 0
    '''
    The threshold used to select the plateaus.
    '''

    def sort_by(self, what="sums", reverse=False):
        '''
        Return an array of indices that orders the plateaus by some value.
        '''
        what = getattr(self, what)
        order = sorted(self.indices, key=lambda i: what[i])
        if reverse:
            order.reverse()
        return numpy.array(order)
        

def plateaus(frame, vthreshold=None):
    '''Label contiguous regions above value threshold.

    Return a Plateaus.

    The vthreshold is compared directly to pixel values.  If not given, it
    default to min + 0.001 * (max-min)

    '''
    from scipy import ndimage

    if vthreshold is None:
        vmin = numpy.min(frame)
        vmax = numpy.max(frame)
        vthreshold = vmin + 0.001 * (vmax-vmin)

    thresh = frame > vthreshold
    labels, nlabels = ndimage.label(thresh)
    labs = numpy.arange(nlabels)+1

    return Plateaus(
        number = nlabels,
        labeled = labels,
        bboxes = ndimage.find_objects(labels, nlabels),
        sums = numpy.array(ndimage.sum_labels(frame, labels, labs)),
        counts = numpy.array(ndimage.sum_labels(thresh, labels, labs)),
        coms = numpy.array(ndimage.center_of_mass(frame, labels, labs)),
        threshold = vthreshold
    )


@dataclasses.dataclass
class SelectActivity:
    '''
    Characterize activity in a subset of a frame
    '''

    selection: numpy.ndarray
    '''
    The array over selected channels
    '''

    channels: list | slice
    '''
    The channels in this selection
    '''

    bln: BaselineNoise
    '''
    Calcualted baseline noise
    '''

    plats: list
    '''
    The plateaus found
    '''

    bbox: tuple
    '''
    A bounding box for all objects found
    '''

    nsigma: float
    '''
    The number of sigma above median for plateau threshold.
    '''

    @property
    def activity(self):
        '''
        The selection array reduced by the median.
        '''
        return self.selection - self.bln.med

    @property
    def thresholded(self):
        '''
        The activity with below threshold values masked
        '''
        act = self.activity
        return ma.array(act, mask = act <= self.plats.threshold)

def select_activity(frame, ch, nsigma=3.0):
    '''Select activity from a "frame" array spanning many channels.

    Given a full frame array, select channel rows given by ch, apply a threshold
    that is nsigma*sigma above median to find bounding box of activity.

    Return the selected frame that has below-threshold pixels masked.

    '''
    plane = frame[ch, :]    # select channels
    bln = baseline_noise(plane)
    thresh = bln.med + nsigma*bln.sigma
    plats = plateaus(plane, thresh)
    if plats.number <= 0:
        raise ValueError(f'no activity in frame {frame.shape=} from {ch=}')
    assert plats.number > 0
    bbox = union_bbox(*plats.bboxes)
    return SelectActivity(
        selection = plane,
        channels = ch,
        bln = bln,
        plats = plats,
        bbox = bbox,
        nsigma = nsigma)
