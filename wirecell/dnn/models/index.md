---
generated: 2026-08-07
source-hash: 79c2503a1a425756
children-hash: acca82aed44e7be0
---

# wirecell/dnn/models

Neural network model definitions for Wire-Cell signal processing, providing U-Net architectures for image segmentation and denoising tasks. The package exposes the classic Ronneberger-Fischer-Brox U-Net, a Vision Transformer bottlenecked variant (`ViTUNetCrossView`) for processing large detector images via multi-view splitting, and `XViewUNet`, which puts time-banded cross-view attention on top of one U-Net trunk per view.

## Modules

| Module | Purpose | Key Symbols |
|---|---|---|
| `unet` | Configurable U-Net implementation with optional batch norm, bilinear upsampling, and padding | `UNet`, `dconv`, `dsamp`, `umerge` |
| `UViTrio` | Development workspace for `ViTUNetCrossView`: a ViT-bottlenecked U-Net with spatial view splitting | `ViTUNetCrossView`, `MultiViewUNet`, `ViTUNetGenerator` |
| `xvunet` | Per-view U-Net trunks plus time-banded cross-view self-attention, fused back at full resolution; the attention path is zero-init gated so a fresh model reproduces the trunks exactly | `XViewUNet`, `BandedAttentionBlock` |

## Dependencies

- `torch`, `torch.nn` — `Conv2d`, `MaxPool2d`, `ConvTranspose2d`, `Upsample`, `Sequential`
- `torch.nn.functional.scaled_dot_product_attention` — banded attention in `xvunet`
- `torch.utils.checkpoint` — activation checkpointing in `xvunet`
- `UViTrio.uvcgan2.models.generator.vitunet_crossview` — `ViTUNetCrossView`
