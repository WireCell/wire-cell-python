---
generated: 2026-04-29
source-hash: 3e9d739b8f727e2f
---

# wirecell/test/

Testing utilities and CLI commands for Wire Cell signal processing validation. The primary focus is the "simple splat / sim+signal" (ssss) comparison test that reproduces signal bias, efficiency, and resolution metrics from the MicroBooNE SP-1 paper. Also includes plot helpers for empirical noise model and noise tools tests.

## Modules

| Module | Purpose | Key Symbols |
|--------|---------|-------------|
| `ssss` | Core splat/signal comparison logic: frame loading, alignment, metrics | `Frame`, `Metrics`, `load_frame`, `align_channel_ranges`, `calc_metrics`, `plot_frames`, `plot_metrics` |
| `empiricalnoisemodel` | Plot spectra and per-channel noise model output from test data | `plot` |
| `noisetools` | Plot wave energies, RMS, spectra, and autocorrelations from noise test data | `NamArr`, `Dat`, `wave_plots`, `plot_proto`, `plot` |
| `__main__` | CLI entry point wiring all test sub-commands | `cli`, `ssss_args` |

## CLI Commands

| Command | Description |
|---------|-------------|
| `plot` | Make plots from file made by `test_<test>` (dispatches to per-module `plot()`). |
| `plot-ssss` | Run splat/signal comparison and produce diagnostic plots per plane. |
| `ssss-metrics` | Compute and write per-plane ssss metrics (bias, resolution, inefficiency) to JSON. |
| `plot-metrics` | Plot per-plane metrics from JSON files produced by `ssss-metrics`. |

## Dependencies

| Import | Role |
|--------|------|
| `wirecell.util.ario` | Archive I/O for loading test data files |
| `wirecell.util.plottools` | Multi-page PDF output via `pages` |
| `wirecell.util.peaks` | `select_activity`, `baseline_noise`, `BaselineNoise`, `gauss` |
| `wirecell.util.functions` | Unit parsing via `unitify_parse`, `unitify` |
| `wirecell.util.cli` | CLI context and logging helpers |
| `wirecell.util.codec` | `json_dumps` for metric output |
| `wirecell.units` | Physical unit constants |
