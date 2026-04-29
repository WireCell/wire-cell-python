---
generated: 2026-04-29
source-hash: 2c0778805863c4ea
---

# wirecell/sigproc/test

Test suite for the `wirecell.sigproc.fwd` forward simulation module. Covers signal construction, convolution, Gaussian waveforms, electronics and field responses, detector responses, and noise generation across known detector configurations.

## Modules

| Module | Purpose | Key Symbols |
|--------|---------|-------------|
| `test-fwd.py` | pytest tests for `wirecell.sigproc.fwd` | `test_signal`, `test_rebin`, `test_convolution`, `test_gauss`, `test_electronics_response`, `test_field_response`, `test_detector_response`, `test_noise` |

## Dependencies

- `wirecell.units` — physical unit constants used in time and frequency parameters
- `wirecell.sigproc.fwd` — module under test: `Signal`, `Convolution`, `Noise`, `ElecResponse`, `FieldResponse`, `DetectorResponse`, `gauss_wave`, `gauss_signal`, `rebin`, `make_times`, `known_responses`
