---
generated: 2026-04-29
source-hash: b0b5f99c8bbca858
children-hash: beec6571a0b0132c
---

# wirecell/dnn/models/UViTrio/uvcgan2

Top-level namespace package for the UViTrio/uvcgan2 image-to-image translation framework. Organizes generator model architectures and supporting PyTorch utilities for unpaired image translation using Vision Transformer bottlenecked U-Net structures.

## Modules

| Module | Purpose | Key Symbols |
|---|---|---|
| `models` | Generator architectures with ViT-bottlenecked U-Net structures | `ViTUNetGenerator`, `ViTUNetCrossView` |
| `torch` | Factory functions for norm layers, activations, optimizers, and losses | `get_norm_layer`, `get_activ_layer`, `select_optimizer`, `select_loss` |

## Dependencies

- `torch.nn` — `Identity`, `LayerNorm`, `BatchNorm2d`, `InstanceNorm2d`, activation and loss modules
- `torch.optim` — `AdamW`, `Adam`
