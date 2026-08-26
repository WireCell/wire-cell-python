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


def test_line_source_response_skips_all_zero_paths():
    """All-zero ('sentinel') paths must not pollute trapezoidal weights.

    Reproduces the PDVD W-plane case where pp=0 is stored with all-zero
    currents; without filtering, that entry pins the central weight at
    zero and undernormalises the collection-plane response by ~12%.
    """
    pitch = 5.0 * units.mm
    cur_real = np.array([1.0, 2.0, 3.0, 0.0])
    paths = [
        PathResponse(current=cur_real, pitchpos=0.5 * pitch, wirepos=0.0),
        # all-zero sentinel at pp=0 — must be skipped:
        PathResponse(current=np.zeros(4),    pitchpos=0.0,        wirepos=0.0),
    ]
    plane = _make_plane(0, pitch, paths)
    got = l1sp.line_source_response(plane)
    # After filtering the sentinel, the only real path (xi=0.5*pitch) is
    # mirrored to xi=-0.5*pitch with the same current, giving two equal
    # samples spanning the wire pitch.  Trapezoidal weights: each half-pitch
    # carries weight 0.5*pitch, so integral = 1.0 * pitch * cur_real, then
    # divided by pitch → cur_real.
    np.testing.assert_allclose(got, cur_real, atol=1e-12)


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


# ---------------------------------------------------------------------------
# er_kind='json' path (PDVD-top style)
# ---------------------------------------------------------------------------

def _write_synthetic_json_er(tmp_path, period_ns=100.0, n=200,
                             peak_idx=20, peak_amp=1.0):
    """Write a minimal JsonElecResponse JSON.bz2 to tmp_path/er.json.bz2.

    Format mirrors what wirecell.sigproc.track_response.load_jsonelec
    expects: {"times":[...], "amplitudes":[...]} where times are at the
    given period.  Amplitudes are a unit-amplitude triangular bump so
    convolving with a known FR is easy to reason about.
    """
    import bz2, json
    times = (np.arange(n) * period_ns).tolist()
    amp = np.zeros(n)
    for i in range(-5, 6):
        amp[peak_idx + i] = peak_amp * (1.0 - abs(i) / 5.0)
    payload = {"times": times, "amplitudes": amp.tolist()}
    p = tmp_path / "er.json.bz2"
    with bz2.open(p, 'wt') as f:
        json.dump(payload, f)
    return str(p)


def test_build_l1sp_kernels_json_er(fake_fr_load, tmp_path, monkeypatch):
    """er_kind='json' loads ER from a JSON.bz2 and produces well-formed kernels."""
    er_path = _write_synthetic_json_er(tmp_path)
    # load_jsonelec resolves via WIRECELL_PATH; point it at tmp_path.
    monkeypatch.setenv("WIRECELL_PATH", str(tmp_path))

    out = l1sp.build_l1sp_kernels(
        fr_file='dummy.json.bz2',
        er_kind='json',
        er_file='er.json.bz2',
    )

    # Schema sanity
    assert out['meta']['elec_type'] == 'json'
    assert out['meta']['gain_mV_per_fC'] is None
    assert out['meta']['shaping_us'] is None
    assert out['meta']['er_file'] == 'er.json.bz2'
    # Kernels are real, finite, and non-trivial.
    for p in out['planes']:
        bip = np.asarray(p['positive']['bipolar'])
        assert np.all(np.isfinite(bip))
        assert np.max(np.abs(bip)) > 0


def test_build_l1sp_kernels_json_er_requires_file(fake_fr_load):
    with pytest.raises(ValueError, match='er_file'):
        l1sp.build_l1sp_kernels(fr_file='dummy.json.bz2', er_kind='json')


def test_build_l1sp_kernels_unknown_er_kind(fake_fr_load):
    with pytest.raises(ValueError, match='er_kind'):
        l1sp.build_l1sp_kernels(fr_file='dummy.json.bz2', er_kind='warm-ish')


# ---------------------------------------------------------------------------
# output_window_us padding (PDVD bipolar tail wraparound fix)
# ---------------------------------------------------------------------------

def test_build_l1sp_kernels_output_window_pads(fake_fr_load):
    """When output_window_us > native FR window, kernels are zero-padded.

    Synthetic FR is 200 samples × 100 ns = 20 µs.  Request 40 µs → expect
    n_samples == 400 in meta and kernel arrays.
    """
    out = l1sp.build_l1sp_kernels(
        fr_file='dummy.json.bz2',
        output_window_us=40.0,
    )
    assert out['meta']['n_samples'] == 400
    assert out['meta']['fr_n_samples_native'] == 200
    assert out['meta']['output_window_us'] == 40.0
    for p in out['planes']:
        for pol in ('positive', 'negative'):
            assert len(p[pol]['bipolar']) == 400
            assert len(p[pol]['unipolar']) == 400


def test_build_l1sp_kernels_output_window_no_op_when_smaller(fake_fr_load):
    """output_window_us shorter than native FR window is silently ignored."""
    out = l1sp.build_l1sp_kernels(
        fr_file='dummy.json.bz2',
        output_window_us=5.0,    # <  20 µs native
    )
    # n_samples falls back to the native length, no fr_n_samples_native key
    # would be useful but we still populate it so callers see a uniform schema.
    assert out['meta']['n_samples'] == 200
    assert out['meta']['fr_n_samples_native'] == 200


def test_build_l1sp_kernels_output_window_unset_no_extra_keys(fake_fr_load):
    """When output_window_us is None, the extra meta keys are omitted so
    legacy PDHD JSONs stay bit-identical."""
    out = l1sp.build_l1sp_kernels(fr_file='dummy.json.bz2')
    assert 'output_window_us' not in out['meta']
    assert 'fr_n_samples_native' not in out['meta']
