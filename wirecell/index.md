---
generated: 2026-04-29
source-hash: ccc39640cf7109d6
children-hash: 0c1f2ca364323f43
---

# wirecell/

Top-level Python package for the Wire-Cell Toolkit, bundling simulation, signal processing, imaging, visualization, and utility sub-packages under a unified `wcpy` CLI. Also provides a Python interface to the `wire-cell` executable, a Wire-Cell configuration object model, and the complete system of physical units used across all sub-packages.

## Modules

| Module | Purpose | Key Symbols |
|--------|---------|-------------|
| `__init__.py` | Package marker | — |
| `__main__.py` | Root `wcpy` CLI: discovers and registers all sub-package CLIs, plus a `summary` command | `cli`, `main`, `cmd_summary` |
| `command.py` | Python interface to the `wire-cell` executable; config load/merge/serialize | `Config`, `WireCell` |
| `units.py` | Complete Wire-Cell system of units derived from `WireCellUtil/Units.h` | `mm`, `cm`, `m`, `ns`, `us`, `MeV`, `eV`, `volt`, `tesla`, `clight`, and all SI/derived constants |

## Sub-packages

| Sub-package | Purpose |
|-------------|---------|
| `aux` | DFT validation and benchmarking against NumPy |
| `bee` | Bee visualization server file loading and inspection |
| `dfp` | Data-flow programming graph construction and GraphViz rendering |
| `dnn` | PyTorch DNN training framework for ROI finding and related tasks |
| `docs` | index.md generation, staleness checking, and LLM prompt emission |
| `gen` | Deposition and frame simulation (morse, line tracks, noise spectra) |
| `img` | Cluster graph loading, VTK/Bee conversion, and blob plotting |
| `ls4gan` | LS4GAN↔WCT frame conversion and array comparison metrics |
| `pgraph` | WCT config → GraphViz dot visualization |
| `plot` | Frame waveform and spectral plotting CLI |
| `pytorch` | Torch-script DFT module generation |
| `raygrid` | Ray-grid tiling algorithm and wire-plane coordinate tools |
| `resp` | Field/electronics response ingestion, resampling, and plotting |
| `sigproc` | Signal processing: field responses, noise models, ForWaRD simulation |
| `test` | ssss comparison tests and noise model validation plots |
| `util` | General-purpose I/O, units, LMN resampling, wire geometry, and plotting helpers |
| `validate` | Magnify ROOT histogram diffing and validation plots |

## CLI Commands

| Command | Description |
|---------|-------------|
| `summary` | Print a one-line summary of every wcpy namespace and its commands |

## Dependencies

| Import | Role |
|--------|------|
| `wirecell.util.cli` | `context`, `log` — shared CLI group factory and logging used by `__main__` |
