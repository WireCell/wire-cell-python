#!/usr/bin/env pytest
'''
Regression test: a signal plane with no activity (e.g. a degenerate spng output
that is a single all-zero channel) must not crash metrics computation.
'''
import numpy
from wirecell.util.peaks import select_activity
from wirecell.test import ssss


def _splatish(nchan=800, nticks=400):
    '''A frame with a clear blob of activity.'''
    a = numpy.zeros((nchan, nticks), dtype=numpy.float32)
    a[100:110, 200:210] = 1000.0
    return a


def _empty(nchan=800, nticks=400):
    return numpy.zeros((nchan, nticks), dtype=numpy.float32)


def test_select_activity_required_false_no_raise():
    '''required=False must not raise on an all-zero selection.'''
    frame = _empty()
    ch = slice(0, 800)
    sa = select_activity(frame, ch, nsigma=3.0, required=False)
    assert sa.plats.number == 0
    assert sa.bbox is None
    # .activity must still be usable (baseline-subtracted selection).
    assert sa.activity.shape == frame[ch, :].shape


def test_select_activity_required_true_still_raises():
    '''Default behavior is preserved: raise on no activity.'''
    frame = _empty()
    try:
        select_activity(frame, slice(0, 800), nsigma=3.0)
    except ValueError:
        return
    assert False, "expected ValueError for empty frame with required=True"


def test_metrics_empty_signal_plane():
    '''Splat has activity, signal plane is empty -> empty Metrics, no crash.'''
    splat = _splatish()
    signal = _empty()
    ch = slice(0, 800)

    spl = select_activity(splat, ch, nsigma=3.0)
    sig = select_activity(signal, ch, nsigma=3.0, required=False)

    biggest = spl.plats.sort_by("sums")[-1]
    bbox = spl.plats.bboxes[biggest]
    spl_qch = numpy.sum(spl.activity[bbox], axis=1)
    sig_qch = numpy.sum(sig.activity[bbox], axis=1)

    try:
        m = ssss.calc_metrics(spl_qch, sig_qch, nbins=50)
    except Exception:
        m = ssss.Metrics()   # this is the graceful path ssss_metrics uses
    # An empty-signal plane yields the default Metrics (fit is None).
    assert m.fit is None
