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
//   adc_tick  - DAQ tick as unit expression
//   chndb_resp - path to chndb-resp.jsonnet for overlay (null = skip overlay)
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
        adc_tick:   "512*ns",
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
        chndb_resp: "pgrapher/experiment/protodunevd/chndb-resp.jsonnet",
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
        chndb_resp: "pgrapher/experiment/protodunevd/chndb-resp.jsonnet",
    },
}
