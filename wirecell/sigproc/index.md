---
generated: 2026-04-29
source-hash: 3c775cac62baef30
children-hash: 269d162c56b018e0
---

# wirecell/sigproc/

Signal processing toolkit for Wire Cell, covering field response loading and conversion, electronics response modeling, noise spectrum handling, and forward simulation (ForWaRD). Provides both a programmatic API and a rich CLI for working with Garfield simulation data, detector field responses, and noise models across multiple detector configurations (MicroBooNE, SBND, ProtoDUNE-HD/VD).

## Modules

| Module | Purpose | Key Symbols |
|--------|---------|-------------|
| `response/` | Field and electronics response math, schema, arrays, persistence, plots | `ResponseFunction`, `electronics`, `convolve`, `rf1dtoschema`, `persist` |
| `noise/` | Noise spectrum schema, loaders, persistence, and plots | `NoiseSpectrum`, `load`, `dump`, `plot_many` |
| `paper/` | Figure generation scripts for signal processing publications | `figure_adc`, `filter_response_functions` |
| `test/` | pytest suite for `fwd` module | `test_signal`, `test_detector_response`, `test_noise` |
| `fwd.py` | ForWaRD signal/noise simulation: Signal, Noise, Convolution classes and plotting | `Signal`, `Noise`, `Convolution`, `DetectorResponse`, `FieldResponse`, `ElecResponse`, `gauss_wave`, `rebin` |
| `garfield.py` | Load and normalize Garfield `.dat` tarball archives into `ResponseFunction` lists | `load`, `toarrays`, `convert` |
| `l1sp.py` | Build L1SPFilterPD response-kernel files from FR + electronics | `build_l1sp_kernels`, `save_l1sp_kernels`, `load_l1sp_kernels`, `negative_half` |
| `track_response.py` | FR⊗ER perpendicular-line track response for wire-plane detectors | `load_detector_config`, `make_track_response`, `make_plot`, `parse_chndb_resp`, `line_source_response` |
| `plots.py` | Matplotlib plotting utilities for field responses, electronics, and digitized waveforms | `plot_digitized_line`, `garfield_exhaustive`, `one_electronics`, `fine_response`, `plane_impact_blocks` |
| `minisim.py` | Minimal drift simulation applying response functions to Geant4 hits | `Minisim`, `Hit` |
| `downsample.py` | Downsampling study utilities for field response convolution | `checkit` |

## CLI Commands

| Command | Description |
|---------|-------------|
| `fr2npz` | Convert field response JSON/JSON.bz2 to NPZ, optionally convolving with electronics response |
| `frzero` | Emit FR with all but selected wire regions zeroed, optionally made uniform |
| `response-info` | Print origin, period, speed, plane locations from a field response file |
| `convert-garfield` | Convert Garfield archive (zip/tar/tgz) to WCT field response JSON |
| `plot-garfield-exhaustive` | Plot all Garfield current responses to a PDF |
| `plot-garfield-track-response` | Plot shaped and digitized perpendicular-track response from Garfield data |
| `plot-response-compare-waveforms` | Overlay waveforms from two field response files |
| `plot-response` | Plot per-plane field responses |
| `plot-response-conductors` | Plot per-conductor (wire/strip) responses |
| `plot-spectra` | Plot per-plane response spectra |
| `plot-electronics-response` | Plot electronics response function for given gain/shaping |
| `convert-noise-spectra` | Convert external noise spectra (MicroBooNE v1, ICARUS) to WCT format |
| `convert-electronics-response` | Convert tabular electronics response to WCT format |
| `plot-noise-spectra` | Plot WCT noise spectra file contents |
| `channel-responses` | Produce per-channel calibrated response JSON from a ROOT TH2D |
| `plot-configured-spectra` | Plot spectra from a configured WCT noise model component |
| `track-response` | Compute and plot FR⊗ER perpendicular-line track response per plane (U, V) |
| `gen-l1sp-kernels` | Build per-detector `L1SPFilterPD` bipolar/unipolar kernel file (JSON+bz2) from a field-response archive |
| `fwd` | Exercise the ForWaRD technique with track/blip signals and noise |

## Dependencies

- `wirecell.units` — physical unit constants used throughout
- `wirecell.util.cli` — `context`, `log`, `jsonnet_loader` for CLI scaffolding
- `wirecell.util.jsio` — JSON/Jsonnet loading and path resolution
- `wirecell.util.fileio` — `wirecell_path`, file source loader
- `wirecell.util.functions` — `unitify` for unit string parsing
- `wirecell.util.plottools` — `pages` context manager for multi-page PDF output
- `wirecell.resp.garfield` — `dataset_asdict` for Garfield data parsing
