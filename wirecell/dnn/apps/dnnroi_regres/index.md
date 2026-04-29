---
generated: 2026-04-29
source-hash: 895f252f59cd11b1
---

# wirecell/dnn/apps/dnnroi_regres/

A DNNROI training application variant that replaces binary ROI classification with a two-headed regression approach. The network produces simultaneous classification and value-regression outputs trained with a composite "hurdle loss" combining binary cross-entropy and MSE. Dataset handling mirrors the standard `dnnroi` app but targets continuous charge-deposition values rather than binary masks.

## Modules

| Module | Purpose | Key Symbols |
|--------|---------|-------------|
| `__init__.py` | App API entry point; defines training components | `Network`, `Dataset`, `Trainer`, `Optimizer`, `Criterion` (hurdle loss) |
| `data.py` | HDF5 dataset loaders for rec and tru frames | `Rec`, `Tru`, `Dataset` |
| `model.py` | Two-output UNet network | `Network` |
| `transforms.py` | Crop, rebin, and normalize frame tensors | `Rec`, `Tru`, `Params`, `DimParams` |

## Dependencies

| Import | Role |
|--------|------|
| `wirecell.dnn.data.hdf` | `Single`, `Multi`, `Domain`, `ReMatcher` — HDF5 dataset abstractions |
| `wirecell.dnn.train.Classifier` | Generic training loop used as `Trainer` |
| `wirecell.dnn.models.unet.UNet` | U-Net backbone (1 input channel, 2 output channels) |
