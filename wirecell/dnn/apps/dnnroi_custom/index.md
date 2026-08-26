---
generated: 2026-04-29
source-hash: 8b2020d017efd18e
---

# wirecell/dnn/apps/dnnroi_custom

A customizable variant of the DNNROI training application for wire-cell signal processing ROI identification. It pairs HDF5 frame-file datasets (conventional sigproc "rec" and truth "tru") with a U-Net model trained via binary cross-entropy and SGD. Configuration of cropping, rebinning, and normalization parameters is exposed at construction time to support detectors beyond the defaults.

## Modules

| Module | Purpose | Key Symbols |
|---|---|---|
| `__init__` | App-API assembly: wires together model, data, trainer, criterion, and optimizer | `Network`, `Dataset`, `Trainer`, `Criterion`, `Optimizer` |
| `data` | HDF5 dataset loaders for rec and tru frame files, with configurable file/path matching | `Rec`, `Tru`, `Dataset` |
| `model` | U-Net wrapper with sigmoid output for binary ROI mask prediction | `Network` |
| `transforms` | Per-sample crop, rebin, and normalize transforms for rec (float) and tru (binary) data | `DimParams`, `Params`, `Rec`, `Tru` |

## Dependencies

| Import | Role |
|---|---|
| `wirecell.dnn.data.hdf` | `Single`, `Multi`, `Domain`, `ReMatcher` — HDF5 file scanning and tensor loading |
| `wirecell.dnn.train.Classifier` | Generic training loop used as `Trainer` |
| `wirecell.dnn.models.unet.UNet` | Backbone U-Net architecture wrapped by `Network` |
