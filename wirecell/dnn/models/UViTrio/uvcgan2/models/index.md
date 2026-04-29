---
generated: 2026-04-29
source-hash: b0b5f99c8bbca858
children-hash: 488cd45472f14085
---

# wirecell/dnn/models/UViTrio/uvcgan2/models

Model definitions for the UViTrio/uvcgan2 image translation framework. This sub-package organizes generator architectures that combine Vision Transformer bottlenecks with U-Net structures for unpaired image-to-image translation tasks.

## Modules

| Module | Purpose | Key Symbols |
|---|---|---|
| `generator` | Generator architectures with ViT-bottlenecked U-Net structures | `ViTUNetGenerator`, `ViTUNetCrossView` |

## Dependencies

- `..torch.layers.transformer.PixelwiseViT` — patch-wise Vision Transformer bottleneck
- `..torch.layers.unet.UNet` — encoder/decoder backbone
- `..torch.layers.crossview.MultiViewUNet`, `SplitAwareBottleneck` — multi-path UNet components
- `..torch.select.get_activ_layer` — activation layer factory
