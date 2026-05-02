#!/usr/bin/env python
'''
Build the L1SPFilterPD response-kernel file from a field-response tarball
plus electronics parameters.

The output is a JSON+bz2 blob consumed by the C++ ``L1SPFilterPD`` and by
the validator script in ``pdhd/nf_plot/track_response_l1sp_kernels.py``.
It contains, per induction plane (U=0, V=1):

  - ``positive``: bipolar = plane bipolar kernel,
                  unipolar = W-plane kernel,
                  unipolar_time_offset_us = (zero-crossing − W-peak) for
                  this plane (W peak then lands at the bipolar zero
                  crossing in the L1 fit).
  - ``negative``: bipolar = same plane bipolar kernel,
                  unipolar = neg-half(bipolar), no shift.

Time axis: kernels[i] is sampled at ``t0_us + i * period_ns/1000`` µs.
``t0_us`` is computed from the field-response (origin/speed) and the
configured coarse/fine offsets, mirroring the C++ ``linterp`` x0
convention.
'''

from collections import defaultdict

import numpy as np

from wirecell import units
from wirecell.sigproc.response import persist as fr_persist
from wirecell.sigproc import response as wc_resp
from wirecell.util import jsio
from wirecell.util.fileio import wirecell_path


def line_source_response(plane):
    '''
    Perpendicular-line-source response on the central wire.

    Per-region symmetrize + non-uniform trapezoidal integration.
    '''
    pitch = plane.pitch
    n_samples = len(plane.paths[0].current)

    by_r = defaultdict(list)
    for path in plane.paths:
        r = int(round(path.pitchpos / pitch))
        xi = path.pitchpos - r * pitch
        by_r[r].append((xi, np.asarray(path.current, dtype=float)))

    integral = np.zeros(n_samples)
    for items in by_r.values():
        sym = {xi: I for xi, I in items}
        for xi in list(sym):
            if abs(xi) > 1e-9 and (-xi) not in sym:
                sym[-xi] = sym[xi]
        xis = sorted(sym)
        n = len(xis)
        w = np.empty(n)
        w[0] = (xis[1] - xis[0]) / 2.0
        w[-1] = (xis[-1] - xis[-2]) / 2.0
        for i in range(1, n - 1):
            w[i] = (xis[i + 1] - xis[i - 1]) / 2.0
        for xi, wi in zip(xis, w):
            integral += wi * sym[xi]
    return integral / pitch


def kernel_from_fr_line(fr_line, ewave_signed, period_ns):
    '''
    L1SPFilterPD::init_resp() in numpy.

    Returns ADC per single electron.  ``ewave_signed`` should already
    include the −postgain × adc_per_mv prefactor.
    '''
    spectrum = np.fft.rfft(fr_line) * np.fft.rfft(ewave_signed) * period_ns
    return np.fft.irfft(spectrum, n=len(fr_line))


def find_zero_crossing(kernel, t_us):
    '''
    Time (µs) of the zero crossing between the dominant and secondary
    extrema of a bipolar kernel (linear interpolation).
    '''
    idx_dom = int(np.argmax(np.abs(kernel)))
    sign_dom = int(np.sign(kernel[idx_dom]))

    opp = np.where(np.sign(kernel) == -sign_dom)[0]
    if not len(opp):
        raise ValueError('kernel is not bipolar — no opposite-sign sample found')
    idx_sec = opp[int(np.argmax(np.abs(kernel[opp])))]

    lo, hi = min(idx_dom, idx_sec), max(idx_dom, idx_sec)
    for i in range(lo, hi):
        if kernel[i] * kernel[i + 1] <= 0:
            dv = kernel[i + 1] - kernel[i]
            if dv == 0:
                return float(t_us[i])
            return float(t_us[i] + (-kernel[i] / dv) * (t_us[i + 1] - t_us[i]))
    raise ValueError('no zero crossing found between the two extrema')


def negative_half(kernel):
    '''Negative-polarity unipolar basis: keep samples < 0, zero the rest.'''
    return np.where(kernel < 0, kernel, 0.0)


def build_l1sp_kernels(fr_file,
                       gain=14.0 * units.mV / units.fC,
                       shaping=2.2 * units.us,
                       postgain=1.2,
                       adc_per_mv=4096 / 2000.0,
                       coarse_time_offset_us=-8.0,
                       fine_time_offset_us=0.0,
                       elec_type='cold',
                       induction_plane_indices=(0, 1),
                       collection_plane_index=2,
                       frame_origin_plane_index=1):
    '''
    Build the L1SPFilterPD kernel dictionary from a field-response file
    plus electronics parameters.

    Returns a POD dict ready to be serialised by ``save_l1sp_kernels``.
    '''
    fr = fr_persist.load(fr_file, paths=wirecell_path())
    period_ns = float(fr.period)
    n_samples = len(fr.planes[0].paths[0].current)

    times = np.arange(n_samples, dtype=float) * period_ns
    er = np.asarray(
        wc_resp.electronics(times, peak_gain=gain, shaping=shaping,
                            elec_type=elec_type),
        dtype=float)

    # Sign convention from L1SPFilterPD::init_resp(): negate so the
    # bipolar induction-plane kernel has the standard "trough then lobe"
    # shape, then convert WC voltage units to ADC counts.
    ewave_signed = -1.0 * postgain * (adc_per_mv / units.mV) * er

    plane_map = {pl.planeid: pl for pl in fr.planes}
    if collection_plane_index not in plane_map:
        raise KeyError(f'collection plane {collection_plane_index} '
                       f'not in FR file (planes={sorted(plane_map)})')
    k_W = kernel_from_fr_line(line_source_response(plane_map[collection_plane_index]),
                              ewave_signed, period_ns)

    intrinsic_toff_us = (fr.origin / fr.speed) / units.us
    t0_us = -intrinsic_toff_us - coarse_time_offset_us + fine_time_offset_us
    t_us = t0_us + np.arange(n_samples) * (period_ns / 1000.0)

    t_pk_W_us = float(t_us[int(np.argmax(np.abs(k_W)))])

    planes = []
    for plane_idx in induction_plane_indices:
        if plane_idx not in plane_map:
            raise KeyError(f'induction plane {plane_idx} not in FR file '
                           f'(planes={sorted(plane_map)})')
        k_bip = kernel_from_fr_line(line_source_response(plane_map[plane_idx]),
                                    ewave_signed, period_ns)
        t_zc_us = find_zero_crossing(k_bip, t_us)
        # W shift such that W peak lands at this plane's zero crossing.
        unipolar_toff_us = float(t_zc_us - t_pk_W_us)

        planes.append({
            'plane_index': int(plane_idx),
            'zero_crossing_us': float(t_zc_us),
            'positive': {
                'bipolar': k_bip.tolist(),
                'unipolar': k_W.tolist(),
                'unipolar_time_offset_us': unipolar_toff_us,
            },
            'negative': {
                'bipolar': k_bip.tolist(),
                'unipolar': negative_half(k_bip).tolist(),
                'unipolar_time_offset_us': 0.0,
            },
        })

    # Global LASSO frame origin: kernel native time corresponding to "source
    # signal at t = 0" in the LASSO fit.  Per Strategy B (DUNE/PDHD), this is
    # the bipolar zero crossing of the configured reference induction plane
    # (typically V).  Both U and V channel fits use this single value as the
    # overall_time_offset; the per-plane geometric difference between U and V
    # arrival is already encoded in each plane's kernel shape.
    by_plane = {p['plane_index']: p for p in planes}
    if frame_origin_plane_index not in by_plane:
        raise KeyError(f'frame_origin_plane_index {frame_origin_plane_index} '
                       f'not in induction_plane_indices={induction_plane_indices}')
    frame_origin_us = float(by_plane[frame_origin_plane_index]['zero_crossing_us'])

    return {
        'meta': {
            'fr_file': fr_file,
            'gain_mV_per_fC': float(gain / (units.mV / units.fC)),
            'shaping_us': float(shaping / units.us),
            'postgain': float(postgain),
            'adc_per_mv': float(adc_per_mv),
            'coarse_time_offset_us': float(coarse_time_offset_us),
            'fine_time_offset_us': float(fine_time_offset_us),
            'period_ns': period_ns,
            't0_us': float(t0_us),
            'n_samples': int(n_samples),
            'elec_type': elec_type,
            'fr_origin': float(fr.origin),
            'fr_speed': float(fr.speed),
            'collection_plane_index': int(collection_plane_index),
            'collection_peak_us': t_pk_W_us,
            'frame_origin_plane_index': int(frame_origin_plane_index),
            'frame_origin_us': frame_origin_us,
        },
        'planes': planes,
    }


def save_l1sp_kernels(data, outpath):
    '''Write the kernel dict as JSON+bz2.'''
    jsio.dump(outpath, data)


def load_l1sp_kernels(path):
    '''Load a kernel JSON+bz2 file via wirecell.util.jsio.'''
    return jsio.load(path, paths=wirecell_path())
