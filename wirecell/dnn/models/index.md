---
generated: 2026-04-29
source-hash: 7e85750726dbd691
children-hash: acca82aed44e7be0
---

# wirecell/dnn/models

Neural network model definitions for Wire-Cell signal processing, providing U-Net architectures for image segmentation and denoising tasks. The package exposes the classic Ronneberger-Fischer-Brox U-Net and a Vision Transformer bottlenecked variant (`ViTUNetCrossView`) for processing large detector images via multi-view splitting.

## Modules

| Module | Purpose | Key Symbols |
|---|---|---|
| `unet` | Configurable U-Net implementation with optional batch norm, bilinear upsampling, and padding | `UNet`, `dconv`, `dsamp`, `umerge` |
| `UViTrio` | Development workspace for `ViTUNetCrossView`: a ViT-bottlenecked U-Net with spatial view splitting | `ViTUNetCrossView`, `MultiViewUNet`, `ViTUNetGenerator` |

## Dependencies

- `torch`, `torch.nn` — `Conv2d`, `MaxPool2d`, `ConvTranspose2d`, `Upsample`, `Sequential`
- `UViTrio.uvcgan2.models.generator.vitunet_crossview` — `ViTUNetCrossView`
