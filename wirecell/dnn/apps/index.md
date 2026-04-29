---
generated: 2026-04-29
source-hash: 741027dbd4d4bafb
children-hash: c08f026f56196cf2
---

# wirecell/dnn/apps/

A collection of DNN training applications for wire-cell signal processing ROI identification. Each app exposes a standard API (`Network`, `Dataset`, `Trainer`, `Criterion`, `Optimizer`) consumed by the generic `wirecell.dnn` training harness, covering binary classification, configurable variants, regression, and vision-transformer architectures.

## Modules

| Module | Purpose | Key Symbols |
|---|---|---|
| `dnnroi` | DNNROI binary classification training pipeline with UNet model and HDF5 frame loaders | `Network`, `Dataset`, `Trainer`, `Criterion`, `Optimizer` |
| `dnnroi_custom` | Customizable DNNROI variant supporting per-detector crop/rebin/normalization configuration | `Network`, `Dataset`, `Trainer`, `Criterion`, `Optimizer` |
| `dnnroi_regres` | Two-headed regression variant using hurdle loss (BCE + MSE) for continuous charge values | `Network`, `Dataset`, `Trainer`, `Criterion`, `Optimizer` |
| `uvitrio` | ROI finding with ViT-UNet cross-view architecture (`UViTrio`) trained on WCT HDF5 frames | `Network`, `Dataset`, `Trainer`, `Criterion`, `Optimizer` |

## Dependencies

| Import | Role |
|---|---|
| `wirecell.dnn.data.hdf` | Shared HDF5 dataset primitives (`Single`, `Multi`, `Domain`, `ReMatcher`) used by all apps |
| `wirecell.dnn.train.Classifier` | Generic supervised training loop aliased as `Trainer` in each app |
| `wirecell.dnn.models.unet.UNet` | U-Net backbone used by `dnnroi`, `dnnroi_custom`, and `dnnroi_regres` |
| `wirecell.dnn.models.ViTUNetCrossView` | Vision-transformer U-Net backbone used by `uvitrio` |
