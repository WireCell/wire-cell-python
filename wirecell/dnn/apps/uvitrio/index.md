---
generated: 2026-04-29
source-hash: 888f558f862b1768
---

# wirecell/dnn/apps/uvitrio

A self-contained DNN application for ROI finding using a ViT-UNet cross-view architecture (UViTrio). Provides the standard app API (`Network`, `Dataset`, `Trainer`, `Criterion`, `Optimizer`) wired to a vision-transformer U-Net model trained on WCT HDF5 frame files.

## Modules

| Module | Purpose | Key Symbols |
|---|---|---|
| `__init__` | App entry point; assembles the standard training API | `Network`, `Dataset`, `Trainer`, `Criterion`, `Optimizer` |
| `model` | Defines the neural network wrapping `ViTUNetCrossView` | `Network` |
| `data` | HDF5 dataset classes pairing reconstructed and truth frames | `Rec`, `Tru`, `Dataset` |
| `transforms` | Data preprocessing (crop, rebin, normalize, threshold) | `Rec`, `Tru`, `Params`, `DimParams` |

## Dependencies

| Import | Role |
|---|---|
| `wirecell.dnn.models.ViTUNetCrossView` | Cross-view vision-transformer U-Net backbone |
| `wirecell.dnn.data.hdf` | HDF5 domain/matcher/dataset base classes (`Single`, `Multi`, `Domain`, `ReMatcher`) |
| `wirecell.dnn.train.Classifier` | Generic classifier training loop used as `Trainer` |
