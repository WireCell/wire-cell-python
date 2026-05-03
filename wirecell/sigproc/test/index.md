---
generated: 2026-05-02
source-hash: 2c0778805863c4ea
---

# wirecell/sigproc/test

Test suite for `wirecell.sigproc` modules. Covers the `fwd` forward simulation module, the `l1sp` L1SPFilterPD kernel builder, and the `track_response` FR⊗ER perpendicular-line response generator.

## Modules

| Module | Purpose | Key Symbols |
|--------|---------|-------------|
| `test-fwd.py` | pytest tests for `wirecell.sigproc.fwd` | `test_signal`, `test_rebin`, `test_convolution`, `test_gauss`, `test_electronics_response`, `test_field_response`, `test_detector_response`, `test_noise` |
| `test_l1sp.py` | pytest tests for `wirecell.sigproc.l1sp`; pure-function tests plus a `build_l1sp_kernels` integration test using a synthetic `FieldResponse` (no FR data file required) | `test_find_zero_crossing_*`, `test_negative_half`, `test_kernel_from_fr_line_*`, `test_line_source_response_synthetic`, `test_build_l1sp_kernels_*` |
| `test_track_response.py` | pytest tests for `wirecell.sigproc.track_response`; per-detector defaults loader, override semantics, `export_chndb_resp`/`parse_chndb_resp` round-trip, and parity with `l1sp.line_source_response` | `test_n_mip_closed_form`, `test_line_source_response_matches_l1sp`, `test_load_detector_config_*`, `test_export_chndb_resp_*` |

## Dependencies

- `wirecell.units` — physical unit constants used in time and frequency parameters
- `wirecell.sigproc.fwd` — module under test by `test-fwd.py`
- `wirecell.sigproc.l1sp` — module under test by `test_l1sp.py`
- `wirecell.sigproc.track_response` — module under test by `test_track_response.py`
- `wirecell.sigproc.response.schema` — `FieldResponse`, `PlaneResponse`, `PathResponse` for synthetic FR construction
- `wirecell.util.jsio` — used to verify exported chndb-resp jsonnet files round-trip cleanly
