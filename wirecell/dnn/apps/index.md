---
generated: 2026-08-07
source-hash: de18e89ed7f499ea
children-hash: 0da6736109b78fc3
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
| `xvunet` | ROI finding from deconvolved images only, via per-view UNet trunks plus time-banded cross-view attention (`XViewUNet`) | `Network`, `Dataset`, `Trainer`, `Criterion`, `Optimizer` |

## Dependencies

| Import | Role |
|---|---|
| `wirecell.dnn.data.hdf` | Shared HDF5 dataset primitives (`Single`, `Multi`, `Domain`, `ReMatcher`) used by all apps |
| `wirecell.dnn.train.Classifier` | Generic supervised training loop aliased as `Trainer` in each app |
| `wirecell.dnn.models.unet.UNet` | U-Net backbone used by `dnnroi`, `dnnroi_custom`, and `dnnroi_regres` |
| `wirecell.dnn.models.ViTUNetCrossView` | Vision-transformer U-Net backbone used by `uvitrio` |
| `wirecell.dnn.models.xvunet.XViewUNet` | Per-view U-Net trunks with cross-view attention, used by `xvunet` |
