---
generated: 2026-04-29
source-hash: 2c1ec87ca85f38e7
children-hash: be524eca4ab8cb69
---

# wirecell/dnn/models/UViTrio/uvcgan2/torch

PyTorch utility layer for the UViTrio/uvcgan2 architecture, providing factory functions for selecting and constructing normalization layers, activation functions, optimizers, and loss functions by name. Serves as the central configuration-driven dispatch mechanism used throughout the model-building submodules. The `layers` sub-package builds on these factories to construct full CNN and transformer-based network components.

## Modules

| Module | Purpose | Key Symbols |
|--------|---------|-------------|
| `select.py` | Factory functions to instantiate norm layers, activations, optimizers, and losses from string/dict config | `get_norm_layer`, `get_norm_layer_fn`, `get_activ_layer`, `select_activation`, `select_optimizer`, `select_loss`, `extract_name_kwargs` |

## Dependencies

- `torch.nn` — `Identity`, `LayerNorm`, `BatchNorm2d`, `InstanceNorm2d`, `GELU`, `ReLU`, `LeakyReLU`, `Tanh`, `Sigmoid`, `L1Loss`, `MSELoss`
- `torch.optim` — `AdamW`, `Adam`
