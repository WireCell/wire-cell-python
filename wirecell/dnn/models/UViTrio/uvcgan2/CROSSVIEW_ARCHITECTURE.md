# Cross-View Architecture

## Overview

The cross-view architecture extends the ViTUNet generator to process three images simultaneously through separate encoder/decoder paths while sharing information at the bottleneck.

## Architecture

### TrioUNetBlock

A single UNet block that handles three parallel image streams:

```
Input: (x0, x1, x2) - three tensors of shape (N, C, H, W)

1. Encode each separately:
   - conv_0(x0) → (y0, r0)  # y0 is downsampled, r0 is skip connection
   - conv_1(x1) → (y1, r1)
   - conv_2(x2) → (y2, r2)

2. Pass to inner module:
   - If inner is TrioUNetBlock: keep streams separate → (y0, y1, y2)
   - If inner is other (e.g., transformer):
     * Concatenate along H: y_concat = (N, C, 3*H, W)
     * Process: y_concat = inner(y_concat)
     * Split back: (y0, y1, y2)

3. Decode each separately with skip connections:
   - deconv_0(y0, r0) → output0
   - deconv_1(y1, r1) → output1
   - deconv_2(y2, r2) → output2

Output: (output0, output1, output2) - three tensors of shape (N, C, H, W)
```

### TrioUNet

Complete UNet with nested TrioUNetBlocks:

- **Input layers**: Three separate conv layers (one per stream)
- **Nested blocks**: Chain of TrioUNetBlocks (one for each level in features_list)
- **Bottleneck**: Shared module that operates on concatenated features (H dimension)
- **Output layers**: Three separate conv layers (one per stream)

### ViTUNetCrossViewGenerator

High-level generator combining TrioUNet with transformer bottleneck:

- Wraps a TrioUNet instance
- Sets PixelwiseViT as the bottleneck (when transformer is integrated)
- Applies output activation to all three streams

## Key Design Decisions

1. **Concatenation dimension**: Always uses **height (H)** dimension for concatenation at the bottleneck
   - Input to bottleneck: (N, C, 3*H, W)
   - This allows the three views to be spatially arranged vertically

2. **Separate parameters**: Each stream has its own encoder/decoder parameters
   - This allows the network to learn view-specific features
   - Only the bottleneck is shared across all three views

3. **Skip connections**: Each stream maintains its own skip connections
   - Ensures proper gradient flow for each view independently

## Usage Example

```python
from uvcgan2.torch.layers.crossview import TrioUNet
import torch.nn as nn

# Create the network
net = TrioUNet(
    features_list=[32, 64, 128],
    activ='relu',
    norm='instance',
    image_shape=(3, 64, 64),  # (C, H, W)
    downsample='conv',
    upsample='upsample-conv',
    rezero=False,
)

# Set a bottleneck (identity for testing, transformer for production)
bottleneck_shape = net.get_inner_shape()  # e.g., (128, 24, 8) for three 8-height views
net.set_bottleneck(nn.Identity())  # or PixelwiseViT(...)

# Forward pass
y0, y1, y2 = net(x0, x1, x2)
```

## Current Status

- ✓ TrioUNetBlock implementation complete
- ✓ TrioUNet implementation complete
- ✓ Basic testing with identity bottleneck working
- ⏳ Transformer integration pending (requires proper shape configuration)

## Future Work

1. **Transformer Integration**: Configure PixelwiseViT to handle concatenated spatial dimensions
2. **Flexible Concatenation**: Consider supporting concatenation along width (W) dimension if needed
3. **Shared Encoders**: Add option to share encoder parameters across views (if desired)
4. **Attention Between Views**: Consider cross-attention mechanisms at intermediate layers
