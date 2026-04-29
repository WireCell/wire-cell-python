---
generated: 2026-04-29
source-hash: fbcfe3bfce528a20
---

# wirecell/dnn/data

PyTorch `Dataset` utilities for loading and splitting WireCell DNN training data, with primary support for HDF5-backed frame files. The package provides flexible regex-based indexing of multi-layer arrays from HDF5 files and a convenience function for reproducible train/eval splits.

## Modules

| Module | Purpose | Key Symbols |
|--------|---------|-------------|
| `__init__` | Dataset splitting utilities | `train_eval_split` |
| `hdf` | HDF5-backed PyTorch datasets for frame data | `ReMatcher`, `Domain`, `Single`, `Multi`, `allkeys` |
| `common` | Shared utilities (placeholder) | — |

## Dependencies

| Dependency | Role |
|-----------|------|
| `torch.utils.data.Dataset` | Base class for `Single` and `Multi` |
| `torch.utils.data.random_split` | Used by `train_eval_split` |
| `h5py` | HDF5 file reading in `hdf.Single` |
| `wirecell.dnn` (logger) | Debug logging throughout `hdf` |
