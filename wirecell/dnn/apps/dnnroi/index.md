---
generated: 2026-04-29
source-hash: 2c2578b6324d9f6e
---

# wirecell/dnn/apps/dnnroi/

The `dnnroi` app implements the DNNROI (Deep Neural Network Region-of-Interest) training pipeline for wire-cell detector data. It pairs a UNet-based binary classifier with HDF5 dataset loaders that read conventional sigproc results (`rec`) and ground-truth ductor ROI frames (`tru`). The module exposes a standard app-level API (Network, Dataset, Trainer, Criterion, Optimizer) consumed by the generic `wirecell.dnn` training harness.

## Modules

| Module | Purpose | Key Symbols |
|---|---|---|
| `__init__` | App API surface wiring together model, data, and training components | `Network`, `Dataset`, `Trainer`, `Criterion`, `Optimizer` |
| `model` | UNet-based binary classifier; applies sigmoid to produce per-pixel ROI probabilities | `Network` |
| `data` | HDF5 dataset loaders for rec (sigproc) and tru (ductor) frame files; paired via `Multi` | `Rec`, `Tru`, `Dataset` |
| `transforms` | Crop, rebin, and threshold transforms applied to rec/tru tensors before training | `DimParams`, `Params`, `Rec`, `Tru` |

## Dependencies

| Import | Role |
|---|---|
| `wirecell.dnn.data.hdf` | HDF5 `Domain`, `ReMatcher`, `Single`, `Multi` dataset primitives |
| `wirecell.dnn.models.unet.UNet` | U-Net encoder–decoder architecture used in `Network` |
| `wirecell.dnn.train.Classifier` | Generic supervised training loop aliased as `Trainer` |
