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
        adc_per_mv: 11.70,
        adc_tick:   "500*ns",
        chndb_resp: "pgrapher/experiment/pdhd/chndb-resp.jsonnet",
    },
    "pdvd-bottom": {
        fr:         "protodunevd_FR_norminal_260324.json.bz2",
        er_kind:    "cold",
        er_file:    null,
        gain:       "7.8*mV/fC",
        shaping:    "2.2*us",
        postgain:   1.1365,
        adc_per_mv: 11.70,
        adc_tick:   "500*ns",
        chndb_resp: "pgrapher/experiment/protodunevd/chndb-resp-bot.jsonnet",
        // PDVD FR file is ~132.5 µs; bipolar induction tail wraps without
        // additional padding. Extend the convolution buffer to 160 µs.
        output_window: "160*us",
    },
    "pdvd-top": {
        fr:         "protodunevd_FR_norminal_260324.json.bz2",
        er_kind:    "json",
        er_file:    "dunevd-coldbox-elecresp-top-psnorm_400.json.bz2",
        gain:       null,
        shaping:    null,
        postgain:   1.52,
        adc_per_mv: 8.192,
        adc_tick:   "500*ns",
        chndb_resp: "pgrapher/experiment/protodunevd/chndb-resp-top.jsonnet",
        // See pdvd-bottom comment.
        output_window: "160*us",
    },
}
