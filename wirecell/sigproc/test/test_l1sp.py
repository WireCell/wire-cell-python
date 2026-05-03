#!/usr/bin/env pytest
"""Tests for wirecell.sigproc.l1sp."""

import numpy as np
import pytest

from wirecell import units
from wirecell.sigproc import l1sp
from wirecell.sigproc.response.schema import (
    FieldResponse, PlaneResponse, PathResponse,
)


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------

def test_find_zero_crossing_exact_sample():
    """Zero crossing falling exactly on a sample is returned as-is."""
    kernel = np.array([+1.0, +0.5, 0.0, -0.5, -1.0, -0.5])
    t_us = np.arange(len(kernel), dtype=float)
    # k[1]*k[2] == 0 → returns t_us[1] + (-0.5/-0.5)*(t_us[2]-t_us[1]) = 2.0
    assert l1sp.find_zero_crossing(kernel, t_us) == pytest.approx(2.0)


def test_find_zero_crossing_interpolated():
    """Linear interpolation between samples of opposite sign."""
    kernel = np.array([+1.0, +0.4, -0.2, -1.0])
    t_us = np.arange(len(kernel), dtype=float)
    # k[1]=0.4, k[2]=-0.2 → crossing at t_us[1] + (0.4/0.6)*1 = 1.6667
    expected = 1.0 + (0.4 / 0.6) * 1.0
    assert l1sp.find_zero_crossing(kernel, t_us) == pytest.approx(expected)


def test_find_zero_crossing_unipolar_raises():
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    t_us = np.arange(len(kernel), dtype=float)
    with pytest.raises(ValueError):
        l1sp.find_zero_crossing(kernel, t_us)


def test_negative_half():
    kernel = np.array([-1.0, 2.0, -3.0, 4.0])
    np.testing.assert_array_equal(
        l1sp.negative_half(kernel),
        np.array([-1.0, 0.0, -3.0, 0.0]),
    )


def test_kernel_from_fr_line_delta():
    """rfft(a) * rfft(δ) → circular conv with delta returns a × period_ns."""
    a = np.array([1.0, 2.0, 3.0, 0.0, -1.0, -2.0, 0.0, 0.0])
    delta = np.zeros_like(a)
    delta[0] = 1.0
    period_ns = 100.0
    got = l1sp.kernel_from_fr_line(a, delta, period_ns)
    np.testing.assert_allclose(got, period_ns * a, atol=1e-10)


def test_kernel_from_fr_line_matches_circular_conv():
    """Match against an explicit circular-convolution reference."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal(16)
    b = rng.standard_normal(16)
    period_ns = 50.0
    got = l1sp.kernel_from_fr_line(a, b, period_ns)
    # Reference: circular convolution via direct full FFT.
    ref = np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b))) * period_ns
    np.testing.assert_allclose(got, ref, atol=1e-10)


def _make_plane(planeid, pitch, paths):
    return PlaneResponse(paths=paths, planeid=planeid,
                         location=0.0, pitch=pitch)


def test_line_source_response_synthetic():
    """Per-region mirror + non-uniform trapezoidal weights, hand-calc."""
    pitch = 5.0 * units.mm
    c0 = np.array([1.0, 0.0, 0.0, 0.0])  # current at xi=0
    c1 = np.array([0.0, 1.0, 0.0, 0.0])  # current at xi=+0.5*pitch
    paths = [
        PathResponse(current=c0, pitchpos=0.0, wirepos=0.0),
        PathResponse(current=c1, pitchpos=0.5 * pitch, wirepos=0.0),
    ]
    plane = _make_plane(0, pitch, paths)

    got = l1sp.line_source_response(plane)

    # Expected weights (per docstring of line_source_response):
    #   xis = [-0.5*pitch, 0, +0.5*pitch] after mirroring xi=+0.5*pitch.
    #   w[0]  = (xis[1]-xis[0])/2 = 0.25*pitch
    #   w[1]  = (xis[2]-xis[0])/2 = 0.5 *pitch
    #   w[2]  = (xis[2]-xis[1])/2 = 0.25*pitch
    # integral = 0.25*pitch * c1 + 0.5*pitch * c0 + 0.25*pitch * c1
    #          = 0.5*pitch*c1 + 0.5*pitch*c0
    # divide by pitch: 0.5*c0 + 0.5*c1
    expected = 0.5 * c0 + 0.5 * c1
    np.testing.assert_allclose(got, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# build_l1sp_kernels integration test (mock FR)
# ---------------------------------------------------------------------------

def _bipolar_current(N, peak_idx):
    """Bipolar pulse: +1 over [peak_idx-3, peak_idx], -1 over (peak_idx, peak_idx+4)."""
    cur = np.zeros(N)
    cur[peak_idx - 3:peak_idx + 1] = 1.0
    cur[peak_idx + 1:peak_idx + 5] = -1.0
    return cur


def _unipolar_current(N, peak_idx):
    cur = np.zeros(N)
    cur[peak_idx - 4:peak_idx + 5] = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1], dtype=float)
    return cur


def _synthetic_fr(N=200, period=100 * units.ns,
                  origin=10.0 * units.cm, speed=1.6 * units.mm / units.us):
    """A 3-plane FR with bipolar U/V and unipolar W."""
    pitch = 5.0 * units.mm
    planes = []
    # Distinct peak indices ensure U and V kernels have different zero crossings.
    for pid, peak in [(0, 30), (1, 40)]:
        cur_a = _bipolar_current(N, peak)
        cur_b = _bipolar_current(N, peak)  # second path so the trapezoidal weights work
        paths = [
            PathResponse(current=cur_a, pitchpos=0.0, wirepos=0.0),
            PathResponse(current=cur_b, pitchpos=0.5 * pitch, wirepos=0.0),
        ]
        planes.append(_make_plane(pid, pitch, paths))
    # W (collection): unipolar
    cur_w_a = _unipolar_current(N, 35)
    cur_w_b = _unipolar_current(N, 35)
    paths_w = [
        PathResponse(current=cur_w_a, pitchpos=0.0, wirepos=0.0),
        PathResponse(current=cur_w_b, pitchpos=0.5 * pitch, wirepos=0.0),
    ]
    planes.append(_make_plane(2, pitch, paths_w))
    return FieldResponse(
        planes=planes, axis=np.array([1.0, 0.0, 0.0]),
        origin=origin, tstart=0.0, period=period, speed=speed,
    )


@pytest.fixture
def fake_fr_load(monkeypatch):
    """Monkeypatch fr_persist.load to return our synthetic FR."""
    fr = _synthetic_fr()
    monkeypatch.setattr(l1sp.fr_persist, 'load',
                        lambda *args, **kwargs: fr)
    return fr


def test_build_l1sp_kernels_basic_structure(fake_fr_load):
    fr = fake_fr_load
    out = l1sp.build_l1sp_kernels(fr_file='dummy.json.bz2')

    assert set(out.keys()) == {'meta', 'planes'}
    assert len(out['planes']) == 2
    assert [p['plane_index'] for p in out['planes']] == [0, 1]

    for p in out['planes']:
        assert set(p['positive']) == {'bipolar', 'unipolar', 'unipolar_time_offset_us'}
        assert set(p['negative']) == {'bipolar', 'unipolar', 'unipolar_time_offset_us'}
        assert p['negative']['unipolar_time_offset_us'] == 0.0
        # negative.unipolar must be the negative-half of the bipolar kernel
        bip = np.asarray(p['positive']['bipolar'])
        neg = np.asarray(p['negative']['unipolar'])
        np.testing.assert_array_equal(neg, l1sp.negative_half(bip))


def test_build_l1sp_kernels_meta_fields(fake_fr_load):
    fr = fake_fr_load
    out = l1sp.build_l1sp_kernels(fr_file='dummy.json.bz2')

    meta = out['meta']
    assert meta['n_samples'] == len(fr.planes[0].paths[0].current)
    assert meta['period_ns'] == pytest.approx(float(fr.period))
    assert meta['collection_plane_index'] == 2
    assert meta['frame_origin_plane_index'] == 1

    # t0_us = -intrinsic_toff_us - coarse_time_offset_us + fine_time_offset_us
    intrinsic_toff_us = (fr.origin / fr.speed) / units.us
    expected_t0 = -intrinsic_toff_us - (-8.0) + 0.0   # defaults
    assert meta['t0_us'] == pytest.approx(expected_t0)


def test_build_l1sp_kernels_frame_origin_us_default(fake_fr_load):
    """Locks commit 3b6dff6: frame_origin_us == V (plane 1) zero crossing."""
    out = l1sp.build_l1sp_kernels(fr_file='dummy.json.bz2')
    by_plane = {p['plane_index']: p for p in out['planes']}
    assert out['meta']['frame_origin_us'] == pytest.approx(
        by_plane[1]['zero_crossing_us'])
    # Sanity: U and V zero crossings differ (peak indices were chosen distinct).
    assert by_plane[0]['zero_crossing_us'] != pytest.approx(
        by_plane[1]['zero_crossing_us'])


def test_build_l1sp_kernels_frame_origin_us_overridden(fake_fr_load):
    """Non-default frame_origin_plane_index picks the right plane."""
    out = l1sp.build_l1sp_kernels(
        fr_file='dummy.json.bz2',
        frame_origin_plane_index=0,
    )
    by_plane = {p['plane_index']: p for p in out['planes']}
    assert out['meta']['frame_origin_plane_index'] == 0
    assert out['meta']['frame_origin_us'] == pytest.approx(
        by_plane[0]['zero_crossing_us'])


def test_build_l1sp_kernels_missing_collection_plane(fake_fr_load):
    with pytest.raises(KeyError):
        l1sp.build_l1sp_kernels(fr_file='dummy.json.bz2',
                                collection_plane_index=99)


def test_build_l1sp_kernels_missing_frame_origin_plane(fake_fr_load):
    """frame_origin_plane_index outside induction_plane_indices raises."""
    with pytest.raises(KeyError):
        l1sp.build_l1sp_kernels(
            fr_file='dummy.json.bz2',
            induction_plane_indices=(0, 1),
            frame_origin_plane_index=2,
        )


def test_build_l1sp_kernels_w_kernel_shared(fake_fr_load):
    """Both induction planes share the same W (collection) unipolar kernel."""
    out = l1sp.build_l1sp_kernels(fr_file='dummy.json.bz2')
    w0 = np.asarray(out['planes'][0]['positive']['unipolar'])
    w1 = np.asarray(out['planes'][1]['positive']['unipolar'])
    np.testing.assert_array_equal(w0, w1)
