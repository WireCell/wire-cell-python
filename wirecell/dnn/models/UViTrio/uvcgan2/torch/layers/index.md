---
generated: 2026-04-29
source-hash: 75ed0741a7462416
---

# wirecell/dnn/models/UViTrio/uvcgan2/torch/layers

PyTorch layer building blocks for the UViTrio/uvcgan2 image-to-image translation architecture. Provides CNN utility functions, transformer-based bottlenecks (PixelwiseViT variants), UNet encoder/decoder blocks, and multi-view parallel processing layers. These components compose into full generator networks via nested UNet structures with optional Vision Transformer bottlenecks.

## Modules

| Module | Purpose | Key Symbols |
|--------|---------|-------------|
| `cnn.py` | Convolution size calculations and factory functions for downsample/upsample layers | `calc_conv_output_size`, `calc_conv_transpose_output_size`, `get_downsample_x2_layer`, `get_upsample_x2_layer` |
| `transformer.py` | Vision Transformer blocks and pixelwise ViT for image-space attention | `PixelwiseViT`, `TransformerBlock`, `TransformerEncoder`, `ViTInput`, `FourierEmbedding`, `ExtendedPixelwiseViT` |
| `unet.py` | UNet encoder/decoder blocks and full UNet with pluggable bottleneck | `UNet`, `UNetBlock`, `UNetEncBlock`, `UNetDecBlock`, `UNetLinearEncoder`, `UNetLinearDecoder` |
| `crossview.py` | Multi-view parallel UNet that concatenates streams at the bottleneck | `MultiViewUNet`, `MultiViewUNetBlock`, `SplitAwareBottleneck` |

## Dependencies

- `..select` (`get_norm_layer`, `get_activ_layer`, `extract_name_kwargs`) — activation/normalization layer factories and name parsing used throughout all submodules
