#!/usr/bin/env python3
"""
Generic FR⊗ER perpendicular-line track response for wire-plane detectors.

Implements the per-region mirror + non-uniform trapezoidal line-source
recipe (Option B), which handles FR files that store impacts on only one
side of the wire centerline per region ("half-stored" convention used by
ub-10-half, dune-garfield-1d565, protodunevd_FR_norminal_260324).

Public API:
  load_detector_config(detector, **overrides) → cfg dict
  make_track_response(fr, cfg)               → (waves, adc_tick)  [adc_tick in WC units]
  make_plot(wave_adc, chndb_ref, *, ...)     → writes PNG
  parse_chndb_resp(path)                     → {'u_resp': arr, 'v_resp': arr}
"""

import os
import bz2
import json
from collections import defaultdict

import numpy as np
from scipy.signal import resample as sp_resample
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from wirecell.sigproc.response import persist
from wirecell.sigproc import response as wc_resp
from wirecell import units
from wirecell.util.fileio import wirecell_path
from wirecell.util.functions import unitify
import wirecell.util.jsio as jsio

_DEFAULTS_FILE = os.path.join(os.path.dirname(__file__), 'track_response_defaults.jsonnet')

PLANE_LABELS = {0: 'U', 1: 'V', 2: 'W'}
CHNDB_KEYS = {'U': 'u_resp', 'V': 'v_resp'}
_CHNDB_TICK = 500.0 * units.ns   # chndb-resp arrays are always sampled at 500 ns


def n_mip(pitch):
    """MIP electrons per wire pitch: 1.8 MeV/cm × pitch_cm × 0.7 / 23.6 eV."""
    return (1.8e6 * (pitch / units.cm) * 0.7) / 23.6


def line_source_response(plane):
    """
    Per-region mirror + non-uniform trapezoidal line-source response.

    For each integer wire region r = round(pitchpos/pitch):
      1. local offset ξ = pitchpos − r·pitch
      2. mirror about ξ=0: for every stored ξ_i ≠ 0 that has no −ξ_i
         counterpart, add −ξ_i with the same current.
      3. sort impacts by ξ, apply non-uniform trapezoidal weights:
           w_0 = (ξ_1−ξ_0)/2, w_i = (ξ_{i+1}−ξ_{i−1})/2, w_N = (ξ_N−ξ_{N−1})/2
      4. accumulate region contribution Σ_i w_i × I(ξ_i)
    Divide the total by plane.pitch.
    """
    pitch = plane.pitch
    N = len(plane.paths[0].current)

    by_r = defaultdict(list)
    for path in plane.paths:
        r  = int(round(path.pitchpos / pitch))
        xi = path.pitchpos - r * pitch
        by_r[r].append((xi, np.asarray(path.current, dtype=float)))

    integral = np.zeros(N)
    for items in by_r.values():
        sym = {xi: I for xi, I in items}
        for xi in list(sym):
            if abs(xi) > 1e-9 and (-xi) not in sym:
                sym[-xi] = sym[xi]
        xis = sorted(sym)
        n = len(xis)
        w = np.empty(n)
        w[0]  = (xis[1] - xis[0]) / 2.0
        w[-1] = (xis[-1] - xis[-2]) / 2.0
        for i in range(1, n - 1):
            w[i] = (xis[i + 1] - xis[i - 1]) / 2.0
        for xi, wi in zip(xis, w):
            integral += wi * sym[xi]
    return integral / pitch


def load_jsonelec(filename, paths=None):
    """Load JsonElecResponse JSON.bz2; return (times, amplitudes), both in WC units."""
    if paths is None:
        paths = wirecell_path()
    full = jsio.resolve(filename, paths)
    with bz2.open(full, 'rb') as fh:
        data = json.load(fh)
    return np.array(data['times'], dtype=float), np.array(data['amplitudes'], dtype=float)


def parse_chndb_resp(path):
    """Load chndb-resp.jsonnet via jsio; return {'u_resp': array, 'v_resp': array}."""
    data = jsio.load(path, paths=wirecell_path())
    return {k: np.array(data[k], dtype=float) for k in ('u_resp', 'v_resp')}


def load_detector_config(detector, **overrides):
    """
    Return a flat config dict for `detector` from track_response_defaults.jsonnet.

    CLI overrides (non-None values) replace the corresponding default.
    Unit-bearing strings (gain, shaping, adc_tick) are converted to WCT
    internal units via unitify().
    """
    defaults = jsio.load(_DEFAULTS_FILE)
    if detector not in defaults:
        raise ValueError(f'unknown detector {detector!r}; choices: {sorted(defaults)}')
    cfg = defaults[detector].copy()
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    for key in ('gain', 'shaping', 'adc_tick'):
        if isinstance(cfg.get(key), str):
            cfg[key] = unitify(cfg[key])
    return cfg


def make_track_response(fr, cfg):
    """
    Compute per-plane ADC waveforms from a loaded FR and a config dict.

    Returns (waves, adc_tick) where waves = {plane_label: ndarray} and
    adc_tick is the DAQ tick in WC units.
    """
    period   = fr.period
    N_fr     = len(fr.planes[0].paths[0].current)
    times    = np.arange(N_fr, dtype=float) * period
    adc_tick = cfg['adc_tick']

    er_kind = cfg.get('er_kind', 'cold')
    if er_kind == 'cold':
        er = np.asarray(
            wc_resp.electronics(times, peak_gain=cfg['gain'],
                                shaping=cfg['shaping'], elec_type='cold'),
            dtype=float)
    elif er_kind == 'json':
        er_t, er_a = load_jsonelec(cfg['er_file'])
        er_period  = er_t[1] - er_t[0]
        er_window  = er_t[-1] + er_period
        er = np.zeros(N_fr)
        if abs(er_period - period) > 1e-6:
            n_resamp = int(round(er_window / period))
            er_resamp = sp_resample(er_a, n_resamp)
            m = min(len(er_resamp), N_fr)
            er[:m] = er_resamp[:m]
        else:
            m = min(len(er_a), N_fr)
            er[:m] = er_a[:m]
    else:
        raise ValueError(f'unknown er_kind {er_kind!r}')

    total      = N_fr * period
    N_adc      = int(round(total / adc_tick))
    postgain   = cfg['postgain']
    adc_per_mv = cfg['adc_per_mv']

    waves = {}
    for pl in fr.planes:
        pid   = pl.planeid
        label = PLANE_LABELS.get(pid, f'plane{pid}')
        if label not in CHNDB_KEYS:
            continue
        n_mip_pl  = n_mip(pl.pitch)
        fr_line   = line_source_response(pl)
        wave_frer = wc_resp.convolve(fr_line, er)
        wave_mv   = -(wave_frer * period * n_mip_pl / units.mV * postgain)
        waves[label] = sp_resample(wave_mv, N_adc) * adc_per_mv

    return waves, adc_tick


def make_plot(wave_adc, chndb_ref, *, tick, plane_label, det_label,
              params_str, outpath):
    """
    Two-panel plot: ADC waveform + |FFT| spectrum.

    tick is the ADC sampling period in WC units. chndb_ref may be None to skip the overlay.
    """
    tick_us    = tick / units.us
    N          = len(wave_adc)
    t_us       = np.arange(N) * tick_us
    freqs_mhz  = np.fft.rfftfreq(N, d=tick_us)
    pk_pos     = wave_adc[np.argmax(wave_adc)]
    pk_neg     = wave_adc[np.argmin(wave_adc)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax = axes[0]
    ax.plot(t_us, wave_adc, 'r-', lw=1.5,
            label=f'FR ⊗ ER  (digitized at {tick / units.ns:.0f} ns)  [{params_str}]')

    if chndb_ref is not None:
        chndb_tick_us = _CHNDB_TICK / units.us
        i_neg_adc     = int(np.argmin(wave_adc))
        i_neg_chndb   = int(np.argmin(chndb_ref))
        t_chndb = (t_us[i_neg_adc]
                   + (np.arange(len(chndb_ref)) - i_neg_chndb) * chndb_tick_us)
        scale         = pk_neg / chndb_ref[i_neg_chndb]
        chndb_scaled  = chndb_ref * scale
        ax.plot(t_chndb, chndb_scaled, 'b--', lw=1.5,
                label=f'chndb-resp.jsonnet  (×{scale:.3g}, aligned at neg. peak)')

    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('time (µs)')
    ax.set_ylabel('ADC')
    ax.set_title(
        f'{det_label}  —  plane {plane_label}  —  full window ({N * tick_us:.1f} µs)\n'
        f'FR ⊗ ER  (MIP perpendicular-line track)  '
        f'peak = {pk_pos:.1f} ADC,  trough = {pk_neg:.1f} ADC'
    )
    ax.legend(fontsize=8, loc='upper left')

    ax = axes[1]
    ax.plot(freqs_mhz, np.abs(np.fft.rfft(wave_adc)), 'r-', lw=1.5, label='FR ⊗ ER')
    if chndb_ref is not None:
        freqs_chndb = np.fft.rfftfreq(len(chndb_scaled), d=chndb_tick_us)
        ax.plot(freqs_chndb, np.abs(np.fft.rfft(chndb_scaled)), 'b--', lw=1.5,
                label='chndb-resp')
    ax.set_xlabel('frequency (MHz)')
    ax.set_ylabel('|FFT| (ADC)')
    ax.set_title('Frequency spectrum')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    return outpath
