---
generated: 2026-04-29
source-hash: 44faf371e3261964
---

# wirecell/resp/

Tools for working with Wire-Cell Toolkit field and electronics responses, including Garfield data ingestion, FR resampling via LMN interpolation, and diagnostic plotting. The package provides both a Python API and a CLI (`wirecell-resp`) for conditioning, resampling, and comparing response files.

## Modules

| Module | Purpose | Key Symbols |
|--------|---------|-------------|
| `garfield.py` | Parse Garfield simulation data files into WCT array format | `dataset_asdict`, `dsdict2arrays`, `parse_text_record` |
| `resample.py` | Resample field responses to a new sampling period using LMN | `resample`, `resample_one`, `rolloff` |
| `plots.py` | Plotting utilities for FR/ER signals in time and frequency domains | `plot_signals`, `plot_paths`, `load_fr`, `eresp`, `multiply_period` |
| `util.py` | Convert FR schema objects to LMN signal lists | `fr2sigs`, `pr2sigs` |
| `__main__.py` | CLI entry point wrapping all commands | `cli` |

## CLI Commands

| Command | Description |
|---------|-------------|
| `gf2npz` | Convert a Garfield data set to a "WCT response NPZ" file |
| `gf-info` | Give info about a garfield dataset |
| `condition` | Condition an FR for resampling (force period, apply roll-off) |
| `resample` | Resample the FR to a new tick period using LMN rationality |
| `compare` | Compare multiple response files in time and frequency domain |
| `lmn-fr-plots` | Make plots for LMN FR presentation |
| `lmn-pdsp-plots` | Generate PDF file with plots illustrating LMN on PDSP |

## Dependencies

| Dependency | Role |
|-----------|------|
| `wirecell.util.lmn` | LMN rational resampling and Signal/Sampling primitives |
| `wirecell.sigproc.response` | FR schema types (`FieldResponse`, `PlaneResponse`, `PathResponse`), persistence, electronics response |
| `wirecell.units` | Unit conversions throughout |
| `wirecell.util.fileio` | `wirecell_path`, source loader |
| `wirecell.util.plottools` | `pages` multi-page PDF helper |
