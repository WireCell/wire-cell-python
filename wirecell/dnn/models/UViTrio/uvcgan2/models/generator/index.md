---
generated: 2026-04-29
source-hash: 71e74c39d800b75b
---

# wirecell/dnn/models/UViTrio/uvcgan2/models/generator

Generator architectures combining Vision Transformer (ViT) bottlenecks with U-Net encoder/decoder structures for image-to-image translation. Provides both a standard single-path ViTUNet generator and a cross-view variant that splits input images into sub-regions processed through separate encoder paths before shared transformer attention.

## Modules

| Module | Purpose | Key Symbols |
|---|---|---|
| `vitunet` | Standard ViT-bottlenecked U-Net generator for image translation | `ViTUNetGenerator` |
| `vitunet_crossview` | Cross-view variant splitting input along a spatial dimension into separate UNet paths | `ViTUNetCrossView` |

## Dependencies

- `...torch.layers.transformer.PixelwiseViT` — patch-wise Vision Transformer used as U-Net bottleneck
- `...torch.layers.unet.UNet` — encoder/decoder backbone consumed by `ViTUNetGenerator`
- `...torch.layers.crossview.MultiViewUNet`, `SplitAwareBottleneck` — multi-path UNet and split-aware bottleneck wrapper consumed by `ViTUNetCrossView`
- `...torch.select.get_activ_layer` — factory for output activation layers
