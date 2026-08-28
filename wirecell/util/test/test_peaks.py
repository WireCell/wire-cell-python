'''
Tests for wirecell.util.peaks.
'''
import numpy
import pytest

from wirecell.util.peaks import baseline_noise


def _cluster(center, n=2000, sigma=0.05, seed=0):
    '''A tight Gaussian cluster of n values around center.'''
    rng = numpy.random.default_rng(seed)
    return rng.normal(center, sigma, n)


def test_baseline_noise_window_centered_on_median():
    '''
    The scalar vrange is an extent about the median, so the fit window must be
    centered on the median -- regardless of how far the median sits from zero.

    Regression: baseline_noise used to add the median twice for scalar vrange,
    centering the histogram on 2*median.  That is invisible when the median is
    ~0 but produces an empty histogram (ValueError) for data whose median is far
    from 0, which crashed `wcpy test plot-ssss`.
    '''
    for center in (0.0, 0.71, -0.4):
        arr = _cluster(center)
        med = float(numpy.median(arr))
        bln = baseline_noise(arr, bins=50, vrange=0.25)

        # Non-empty histogram: the window actually covered the data.
        assert numpy.sum(bln.hist[0]) > 0

        # The histogram window is centered on the median, not 2*median.
        edges = bln.hist[1]
        window_center = 0.5 * (edges[0] + edges[-1])
        assert window_center == pytest.approx(med, abs=0.05)


def test_baseline_noise_nonzero_median_would_regress():
    '''The specific failure mode: median far from 0 must not empty the histogram.'''
    arr = _cluster(0.71)
    bln = baseline_noise(arr, bins=50, vrange=0.25)
    assert numpy.sum(bln.hist[0]) > 0
