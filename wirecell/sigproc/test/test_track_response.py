#!/usr/bin/env pytest
"""Tests for wirecell.sigproc.track_response."""

import numpy as np
import pytest

from wirecell import units
from wirecell.sigproc import track_response as tr
from wirecell.sigproc import l1sp
from wirecell.sigproc.response.schema import PlaneResponse, PathResponse
from wirecell.util import jsio


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_n_mip_closed_form():
    pitch = 5.0 * units.mm
    expected = (1.8e6 * 0.5 * 0.7) / 23.6
    assert tr.n_mip(pitch) == pytest.approx(expected)


def test_line_source_response_matches_l1sp():
    """track_response and l1sp duplicate this implementation; keep them in sync."""
    pitch = 5.0 * units.mm
    paths = [
        PathResponse(current=np.array([1.0, 0.0, 0.0, 0.0]),
                     pitchpos=0.0, wirepos=0.0),
        PathResponse(current=np.array([0.0, 1.0, 0.0, 0.0]),
                     pitchpos=0.5 * pitch, wirepos=0.0),
    ]
    plane = PlaneResponse(paths=paths, planeid=0,
                          location=0.0, pitch=pitch)
    np.testing.assert_array_equal(
        tr.line_source_response(plane),
        l1sp.line_source_response(plane),
    )


# ---------------------------------------------------------------------------
# load_detector_config
# ---------------------------------------------------------------------------

_DETECTORS = ["uboone", "sbnd", "pdhd", "pdvd-bottom", "pdvd-top"]


@pytest.mark.parametrize("name", _DETECTORS)
def test_load_detector_config_required_keys(name):
    cfg = tr.load_detector_config(name)
    for key in ('fr', 'er_kind', 'postgain', 'adc_per_mv',
                'adc_tick', 'chndb_resp'):
        assert key in cfg, f"{name}: missing {key!r}"


@pytest.mark.parametrize("name", _DETECTORS)
def test_load_detector_config_adc_tick_500ns(name):
    """All detectors converged at 500 ns post-resampler (commits 027c308, 2f61318)."""
    cfg = tr.load_detector_config(name)
    assert cfg['adc_tick'] == pytest.approx(500.0 * units.ns)


@pytest.mark.parametrize("name", ["pdvd-bottom", "pdvd-top"])
def test_load_detector_config_pdvd_output_window(name):
    """PDVD induction tail wraparound fix (commit 2710c41)."""
    cfg = tr.load_detector_config(name)
    assert cfg['output_window'] == pytest.approx(160.0 * units.us)


def test_load_detector_config_override_unitified():
    cfg = tr.load_detector_config("uboone", gain="10*mV/fC")
    assert cfg['gain'] == pytest.approx(10.0 * units.mV / units.fC)


def test_load_detector_config_none_override_keeps_default():
    """A None override must NOT replace the default."""
    cfg_default = tr.load_detector_config("uboone")
    cfg_with_none = tr.load_detector_config("uboone", gain=None)
    assert cfg_with_none['gain'] == cfg_default['gain']


def test_load_detector_config_unknown_detector():
    with pytest.raises(ValueError) as excinfo:
        tr.load_detector_config("nonsense")
    msg = str(excinfo.value)
    assert "nonsense" in msg
    # Error message lists known choices.
    assert "uboone" in msg


# ---------------------------------------------------------------------------
# export_chndb_resp ↔ parse_chndb_resp round-trip
# ---------------------------------------------------------------------------

def test_export_chndb_resp_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    waves = {
        'U': rng.standard_normal(64) * 100.0,
        'V': rng.standard_normal(64) * 100.0,
    }
    cfg = {
        'fr':         'fake_fr.json.bz2',
        'er_kind':    'cold',
        'gain':       14.0 * units.mV / units.fC,
        'shaping':    2.2 * units.us,
        'postgain':   1.0,
        'adc_per_mv': 2.048,
        'adc_tick':   500.0 * units.ns,
    }
    out = tmp_path / "chndb-resp-roundtrip.jsonnet"
    tr.export_chndb_resp(waves, cfg, str(out), detector='uboone')

    assert out.exists()
    parsed = tr.parse_chndb_resp(str(out))
    assert set(parsed) == {'u_resp', 'v_resp'}
    # `: .6e` format has ~6 significant digits.
    np.testing.assert_allclose(parsed['u_resp'], waves['U'], rtol=1e-5)
    np.testing.assert_allclose(parsed['v_resp'], waves['V'], rtol=1e-5)


def test_export_chndb_resp_jsonnet_parseable(tmp_path):
    """The emitted file is valid jsonnet (jsio.load decodes it)."""
    waves = {'U': np.arange(8, dtype=float),
             'V': np.arange(8, dtype=float) * -1.0}
    cfg = {
        'fr':         'fake_fr.json.bz2',
        'er_kind':    'cold',
        'gain':       14.0 * units.mV / units.fC,
        'shaping':    2.2 * units.us,
        'postgain':   1.0,
        'adc_per_mv': 1.0,
        'adc_tick':   500.0 * units.ns,
    }
    out = tmp_path / "chndb-resp-tiny.jsonnet"
    tr.export_chndb_resp(waves, cfg, str(out), detector='pdhd')

    data = jsio.load(str(out))
    assert isinstance(data, dict)
    assert set(data) == {'u_resp', 'v_resp'}
    assert len(data['u_resp']) == 8
    assert len(data['v_resp']) == 8
