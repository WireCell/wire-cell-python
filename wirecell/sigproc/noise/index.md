---
generated: 2026-04-29
source-hash: 138c47ecdc399f90
---

# wirecell/sigproc/noise/

Provides noise spectrum data schema, loaders, persistence, and plotting utilities for Wire-Cell signal processing. Supports incoherent and coherent noise models for detectors including MicroBooNE and ICARUS. Noise spectra are parameterized by sampling period, gain, shaping time, wire plane, and wire length.

## Modules

| Module | Purpose | Key Symbols |
|--------|---------|-------------|
| `schema` | Named tuple definitions for noise spectrum data structures | `NoiseSpectrum`, `NoiseSpectrum_v2` |
| `persist` | Serialize/deserialize noise spectra to/from JSON (.json, .json.bz2, .json.gz) | `load`, `dump`, `loads`, `dumps` |
| `microboone` | MicroBooNE-specific noise spectrum loader; handles implicit U/V plane equivalence in v1 format | `load_noise_spectra_v1` |
| `icarus` | ICARUS noise spectrum loaders for incoherent and coherent noise formats | `load_noise_spectra`, `load_coherent_noise_spectra` |
| `plots` | Multi-page PDF plots of noise spectra with frequency/amplitude axes | `plot_many` |

## Dependencies

- `wirecell.units` — physical unit constants (mV, cm, MHz, etc.)
- `wirecell.util` — `unitify` for parsing unit strings
- `wirecell.util.plottools` — `pages` context manager for multi-page PDF output
