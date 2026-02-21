#!/usr/bin/env python
"""
Manual test to verify splitting logic by directly checking tensor shapes.
"""

import torch
import sys
from torch import nn

sys.path.insert(0, '.')

from uvcgan2.torch.layers.crossview import MultiViewUNet, SplitAwareBottleneck


def test_manual_computation():
    """Manually trace through to understand the actual sizes"""
    print("="*60)
    print("Manual computation test")
    print("="*60)

    # Create network
    net = MultiViewUNet(
        num_views=3,
        features_list=[32, 64],
        activ='relu',
        norm='instance',
        image_shape=(3, 2560, 64),
        downsample='conv',
        upsample='upsample-conv',
        rezero=False,
        split_sizes=[800, 800, 960],
        split_dim=2
    )

    # Create input views with actual split sizes
    views = [
        torch.randn(2, 3, 800, 64),
        torch.randn(2, 3, 800, 64),
        torch.randn(2, 3, 960, 64)
    ]

    print(f"\nInput view shapes: {[v.shape for v in views]}")

    # Pass through input layers to see what comes out
    y_list = [net.input_layers[i](views[i]) for i in range(3)]
    print(f"After input layers: {[y.shape for y in y_list]}")

    # Get the innermost block
    innermost = net.get_innermost_block()
    print(f"\nInnermost block encoders:")

    # Manually encode each view through the encoder chain
    curr_list = y_list
    for level_idx, block in enumerate([net.unet, net.unet.get_inner_module()]):
        if not hasattr(block, 'encoders'):
            break
        print(f"\nLevel {level_idx}:")
        print(f"  Input shapes: {[x.shape for x in curr_list]}")

        # Encode
        encoded = [block.encoders[i](curr_list[i]) for i in range(3)]
        y_list_level = [y for y, r in encoded]
        print(f"  After encoding: {[y.shape for y in y_list_level]}")

        # Concatenate
        y_concat = torch.cat(y_list_level, dim=2)
        print(f"  After concatenation: {y_concat.shape}")

        # This is what we need to split!
        if block.inner_module is None or not hasattr(block.inner_module, 'encoders'):
            print(f"\n  ** This is the bottleneck level **")
            print(f"  Concatenated shape at bottleneck: {y_concat.shape}")
            print(f"  We need split_sizes that sum to: {y_concat.shape[2]}")

            # Compute the actual split sizes at this level
            actual_split_sizes = [y.shape[2] for y in y_list_level]
            print(f"  Actual encoded sizes for each view: {actual_split_sizes}")
            print(f"  Sum: {sum(actual_split_sizes)}")
            return actual_split_sizes, y_concat.shape

        curr_list = y_list_level


if __name__ == '__main__':
    actual_splits, bottleneck_shape = test_manual_computation()

    print("\n" + "="*60)
    print("Result:")
    print(f"  Bottleneck split sizes should be: {actual_splits}")
    print(f"  Full bottleneck shape: {bottleneck_shape}")
    print("="*60)
