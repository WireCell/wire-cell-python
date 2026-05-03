// Per-detector defaults for the `wirecell-sigproc track-response` subcommand.
//
// Fields:
//   fr        - FR json.bz2 file (resolved via WIRECELL_PATH)
//   er_kind   - "cold" (parametric wirecell.sigproc.response.electronics) or
//               "json" (JsonElecResponse time-series json.bz2)
//   er_file   - ER file for er_kind="json"; null for er_kind="cold"
//   gain      - FE gain as unit expression (er_kind="cold" only)
//   shaping   - shaping time as unit expression (er_kind="cold" only)
//   postgain  - dimensionless post-shaping gain factor
//   adc_per_mv - ADC counts per mV
//   adc_tick  - sampling tick of the produced waveform (500 ns for all detectors;
//               PDHD and PDVD-bottom have a 512→500 ns resampler upstream so NF
//               and chndb-resp comparison always run at 500 ns post-resampler)
//   chndb_resp - path to chndb-resp.jsonnet for overlay (null = skip overlay)
//   output_window - optional working time window for FR⊗ER convolution.
//               When set and longer than the FR file's native window, FR/ER
//               are zero-padded to this length to avoid circular-convolution
//               wraparound on long bipolar tails (e.g. PDVD induction).
{
    uboone: {
        fr:         "ub-10-half.json.bz2",
        er_kind:    "cold",
        er_file:    null,
        gain:       "14.0*mV/fC",
        shaping:    "2.2*us",
        postgain:   1.2,
        adc_per_mv: 2.048,
        adc_tick:   "500*ns",
        chndb_resp: "pgrapher/experiment/uboone/chndb-resp.jsonnet",
    },
    sbnd: {
        fr:         "garfield-sbnd-v1.json.bz2",
        er_kind:    "cold",
        er_file:    null,
        gain:       "14.0*mV/fC",
        shaping:    "2.0*us",
        postgain:   1.0,
        adc_per_mv: 2.275,
        adc_tick:   "500*ns",
        chndb_resp: "pgrapher/experiment/sbnd/chndb-resp.jsonnet",
    },
    pdhd: {
        fr:         "dune-garfield-1d565.json.bz2",
        er_kind:    "cold",
        er_file:    null,
        gain:       "14.0*mV/fC",
        shaping:    "2.2*us",
        postgain:   1.0,
        // 14-bit ADC, 0.2 V – 1.6 V fullscale → 16384 / 1400 mV.
        adc_per_mv: 16384.0 / 1400.0,
        adc_tick:   "500*ns",
        chndb_resp: "pgrapher/experiment/pdhd/chndb-resp.jsonnet",
    },
    "pdvd-bottom": {
        fr:         "protodunevd_FR_imbalance3p_260501.json.bz2",
        er_kind:    "cold",
        er_file:    null,
        gain:       "7.8*mV/fC",
        shaping:    "2.2*us",
        // PROVISIONAL: postgain absorbs the ~12% W-plane FR
        // under-normalisation caused by an all-zero "sentinel" path at
        // pp=0 in protodunevd_FR_imbalance3p_260501.json.bz2 (calibration
        // is done through the W collection plane).  PDVD-bottom shares
        // electronics with PDHD (postgain = 1.0); 1.1365 / 1.0 ≈ 1.137
        // tracks the FR deficit (peak ×1.124, integral ×1.117).  When
        // the corrected FR lands, drop postgain to 1.0 and regenerate
        // wire-cell-data/pdvd_bottom_l1sp_kernels.json.bz2.  See
        // pdvd/sp_plot/README.md and pdvd/nf_plot/track_response_tool.md
        // in wcp-porting-validation for the full follow-up checklist.
        postgain:   1.1365,
        // Same 14-bit / 1.4 V chip as PDHD bottom electronics.
        adc_per_mv: 16384.0 / 1400.0,
        adc_tick:   "500*ns",
        chndb_resp: "pgrapher/experiment/protodunevd/chndb-resp-bot.jsonnet",
        // PDVD FR file is ~132.5 µs; bipolar induction tail wraps without
        // additional padding. Extend the convolution buffer to 160 µs.
        output_window: "160*us",
    },
    "pdvd-top": {
        fr:         "protodunevd_FR_imbalance3p_260501.json.bz2",
        er_kind:    "json",
        er_file:    "dunevd-coldbox-elecresp-top-psnorm_400.json.bz2",
        gain:       null,
        shaping:    null,
        // PROVISIONAL: same FR-driven over-correction as pdvd-bottom.
        // When the corrected FR lands, scale postgain down by ~12%
        // (≈ 1.52 / 1.117 ≈ 1.36); confirm against a fresh calibration.
        postgain:   1.52,
        adc_per_mv: 8.192,
        adc_tick:   "500*ns",
        chndb_resp: "pgrapher/experiment/protodunevd/chndb-resp-top.jsonnet",
        // See pdvd-bottom comment.
        output_window: "160*us",
    },
}
