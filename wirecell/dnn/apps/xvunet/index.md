---
generated: 2026-08-07
source-hash: 81d8baf8a41618cb
---

# wirecell/dnn/apps/xvunet/

The `xvunet` app trains `XViewUNet`: per-view UNet trunks followed by time-banded cross-view self-attention, for DNNROI-style ROI finding from deconvolved images alone (no MP2/MP3 coincidence-mask input channels). It exposes the standard app-level API (Network, Dataset, Trainer, Criterion, Optimizer) consumed by the generic `wirecell.dnn` harness, reading strictly-parallel per-view HDF5 frame files. Because the model's cross-view branch is gated by zero-initialised scalars, training requires an Adam-family optimizer and bfloat16 autocast; see `wirecell.dnn.models.xvunet` for why both are load-bearing.

## Modules

| Module | Purpose | Key Symbols |
|---|---|---|
| `__init__` | App API surface; builds the optimizer named by `[optimizer] name` (adamw default) with per-optimizer default learning rates | `Network`, `Dataset`, `Trainer`, `Criterion`, `Optimizer` |
| `model` | Adapts `XViewUNet` to the app API, washing INI-string config values into constructor arguments | `Network` |
| `data` | Per-view rec/tru HDF5 loaders concatenated channel-wise into one tensor; verifies the views index identical (file, sample) IDs | `Rec`, `Tru`, `Dataset` |
| `transforms` | Crop, rebin, normalise and threshold transforms applied to rec/tru tensors (same scheme as `dnnroi_custom`) | `DimParams`, `Params`, `Rec`, `Tru` |

## Dependencies

| Import | Role |
|---|---|
| `wirecell.dnn.models.xvunet.XViewUNet` | The cross-view model wrapped by `Network` |
| `wirecell.dnn.data.hdf` | HDF5 `Domain`, `ReMatcher`, `Single` dataset primitives |
| `wirecell.dnn.train.Classifier` | Generic supervised training loop aliased as `Trainer` |
