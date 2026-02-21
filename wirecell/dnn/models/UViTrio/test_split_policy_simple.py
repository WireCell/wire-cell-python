#!/usr/bin/env python
"""
Simplified test script for the new splitting policy functionality.
Uses identity bottleneck to test splitting logic without transformer complexity.
"""

import torch
import sys
from torch import nn

# Add parent directory to path for imports
sys.path.insert(0, '.')

from uvcgan2.torch.layers.crossview import MultiViewUNet, SplitAwareBottleneck


def test_unequal_splits_h():
    """Test unequal splits along H dimension (dim=2) with identity bottleneck"""
    print("\n" + "="*60)
    print("Test 1: Unequal splits along H (dim=2) - Identity bottleneck")
    print("="*60)

    x = torch.randn(2, 3, 2560, 64)  # (N, C, H, W)
    print(f"Input shape: {x.shape}")

    # Create MultiViewUNet
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

    # Get the inner shape (after downsampling)
    inner_shape = net.get_inner_shape()
    print(f"Inner shape (at bottleneck): {inner_shape}")

    # Compute actual bottleneck split sizes by encoding sample views
    # This is the correct way since encoders downsample each view independently
    with torch.no_grad():
        sample_views = [
            torch.randn(1, 3, 800, 64),
            torch.randn(1, 3, 800, 64),
            torch.randn(1, 3, 960, 64)
        ]
        # Pass through input layers
        y_list = [net.input_layers[i](sample_views[i]) for i in range(3)]
        # Encode through all levels to get to bottleneck
        curr_list = y_list
        block = net.unet
        while hasattr(block, 'encoders'):
            encoded = [block.encoders[i](curr_list[i]) for i in range(3)]
            curr_list = [y for y, r in encoded]
            if block.inner_module is None or not hasattr(block.inner_module, 'encoders'):
                break
            block = block.inner_module

        bottleneck_split_sizes = [y.shape[2] for y in curr_list]  # H dimension sizes

    print(f"Bottleneck split sizes (computed from actual encoding): {bottleneck_split_sizes}")
    print(f"Sum of bottleneck split sizes: {sum(bottleneck_split_sizes)}")

    # Create a simple identity transformer with the expected shape
    class IdentityTransformer(nn.Module):
        def __init__(self, image_shape):
            super().__init__()
            self.image_shape = image_shape

        def forward(self, x):
            return x

    identity = IdentityTransformer(inner_shape)
    bottleneck = SplitAwareBottleneck(identity, bottleneck_split_sizes, split_dim=2)
    net.set_bottleneck(bottleneck)

    # Split input and process
    views = list(torch.split(x, [800, 800, 960], dim=2))
    print(f"Split view shapes: {[v.shape for v in views]}")

    outputs = net(views)
    print(f"Output view shapes: {[o.shape for o in outputs]}")

    # Concatenate results
    y = torch.cat(outputs, dim=2)
    print(f"Final output shape: {y.shape}")

    assert y.shape == (2, 3, 2560, 64), f"Expected (2, 3, 2560, 64), got {y.shape}"
    print("✓ Test passed!")


def test_splits_w():
    """Test splits along W dimension (dim=3) with identity bottleneck"""
    print("\n" + "="*60)
    print("Test 2: Splits along W (dim=3) - Identity bottleneck")
    print("="*60)

    x = torch.randn(2, 3, 64, 2560)  # (N, C, H, W)
    print(f"Input shape: {x.shape}")

    # Create MultiViewUNet
    net = MultiViewUNet(
        num_views=3,
        features_list=[32, 64],
        activ='relu',
        norm='instance',
        image_shape=(3, 64, 2560),
        downsample='conv',
        upsample='upsample-conv',
        rezero=False,
        split_sizes=[800, 800, 960],
        split_dim=3
    )

    # Get the inner shape
    inner_shape = net.get_inner_shape()
    print(f"Inner shape (at bottleneck): {inner_shape}")

    # Compute scaled split sizes
    scaling_factor = inner_shape[2] / sum([800, 800, 960])  # W dimension
    bottleneck_split_sizes = [int(size * scaling_factor) for size in [800, 800, 960]]

    print(f"Bottleneck split sizes (after downsampling): {bottleneck_split_sizes}")

    class IdentityTransformer(nn.Module):
        def __init__(self, image_shape):
            super().__init__()
            self.image_shape = image_shape

        def forward(self, x):
            return x

    identity = IdentityTransformer(inner_shape)
    bottleneck = SplitAwareBottleneck(identity, bottleneck_split_sizes, split_dim=3)
    net.set_bottleneck(bottleneck)

    # Split input and process
    views = list(torch.split(x, [800, 800, 960], dim=3))
    print(f"Split view shapes: {[v.shape for v in views]}")

    outputs = net(views)
    print(f"Output view shapes: {[o.shape for o in outputs]}")

    # Concatenate results
    y = torch.cat(outputs, dim=3)
    print(f"Final output shape: {y.shape}")

    assert y.shape == (2, 3, 64, 2560), f"Expected (2, 3, 64, 2560), got {y.shape}"
    print("✓ Test passed!")


def test_equal_splits():
    """Test equal splits"""
    print("\n" + "="*60)
    print("Test 3: Equal splits - Identity bottleneck")
    print("="*60)

    x = torch.randn(2, 3, 192, 64)
    print(f"Input shape: {x.shape}")

    net = MultiViewUNet(
        num_views=3,
        features_list=[32, 64],
        activ='relu',
        norm='instance',
        image_shape=(3, 192, 64),
        downsample='conv',
        upsample='upsample-conv',
        rezero=False,
        split_sizes=[64, 64, 64],
        split_dim=2
    )

    inner_shape = net.get_inner_shape()
    print(f"Inner shape (at bottleneck): {inner_shape}")

    scaling_factor = inner_shape[1] / sum([64, 64, 64])
    bottleneck_split_sizes = [int(size * scaling_factor) for size in [64, 64, 64]]

    print(f"Bottleneck split sizes: {bottleneck_split_sizes}")

    class IdentityTransformer(nn.Module):
        def __init__(self, image_shape):
            super().__init__()
            self.image_shape = image_shape

        def forward(self, x):
            return x

    identity = IdentityTransformer(inner_shape)
    bottleneck = SplitAwareBottleneck(identity, bottleneck_split_sizes, split_dim=2)
    net.set_bottleneck(bottleneck)

    views = list(torch.split(x, [64, 64, 64], dim=2))
    outputs = net(views)
    y = torch.cat(outputs, dim=2)
    print(f"Final output shape: {y.shape}")

    assert y.shape == (2, 3, 192, 64), f"Expected (2, 3, 192, 64), got {y.shape}"
    print("✓ Test passed!")


def test_two_views():
    """Test with 2 views"""
    print("\n" + "="*60)
    print("Test 4: Two views with unequal splits")
    print("="*60)

    x = torch.randn(2, 3, 1000, 64)
    print(f"Input shape: {x.shape}")

    net = MultiViewUNet(
        num_views=2,
        features_list=[32, 64],
        activ='relu',
        norm='instance',
        image_shape=(3, 1000, 64),
        downsample='conv',
        upsample='upsample-conv',
        rezero=False,
        split_sizes=[600, 400],
        split_dim=2
    )

    inner_shape = net.get_inner_shape()
    print(f"Inner shape (at bottleneck): {inner_shape}")

    scaling_factor = inner_shape[1] / sum([600, 400])
    bottleneck_split_sizes = [int(size * scaling_factor) for size in [600, 400]]

    print(f"Bottleneck split sizes: {bottleneck_split_sizes}")

    class IdentityTransformer(nn.Module):
        def __init__(self, image_shape):
            super().__init__()
            self.image_shape = image_shape

        def forward(self, x):
            return x

    identity = IdentityTransformer(inner_shape)
    bottleneck = SplitAwareBottleneck(identity, bottleneck_split_sizes, split_dim=2)
    net.set_bottleneck(bottleneck)

    views = list(torch.split(x, [600, 400], dim=2))
    outputs = net(views)
    y = torch.cat(outputs, dim=2)
    print(f"Final output shape: {y.shape}")

    assert y.shape == (2, 3, 1000, 64), f"Expected (2, 3, 1000, 64), got {y.shape}"
    print("✓ Test passed!")


if __name__ == '__main__':
    print("Testing splitting policy functionality with identity bottleneck")
    print("="*60)

    try:
        test_unequal_splits_h()
        test_splits_w()
        test_equal_splits()
        test_two_views()

        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
