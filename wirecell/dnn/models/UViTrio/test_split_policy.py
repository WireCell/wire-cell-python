#!/usr/bin/env python
"""
Test script for the new splitting policy functionality in ViTUNetCrossView.

Tests:
1. Unequal splits along H (dim=2)
2. Splits along W (dim=3)
3. Equal splits (backward-compatible behavior)
4. Shape validation (should fail)
"""

import torch
import sys

# Add parent directory to path for imports
sys.path.insert(0, '.')

from uvcgan2.models.generator.vitunet_crossview import ViTUNetCrossView


def test_unequal_splits_h():
    """Test unequal splits along H dimension (dim=2)"""
    print("\n" + "="*60)
    print("Test 1: Unequal splits along H (dim=2)")
    print("="*60)

    x = torch.randn(2, 3, 2560, 64)  # (N, C, H, W)
    print(f"Input shape: {x.shape}")

    gen = ViTUNetCrossView(
        features=128,
        n_heads=4,
        n_blocks=2,
        ffn_features=256,
        embed_features=64,
        activ='gelu',
        norm='layer',
        input_shape=(3, 2560, 64),
        output_shape=(3, 2560, 64),
        unet_features_list=[32, 64],
        unet_activ='relu',
        unet_norm='instance',
        split_sizes=[800, 800, 960],
        split_dim=2,  # H dimension
        activ_output='tanh'
    )

    print(f"Model created with split_sizes=[800, 800, 960], split_dim=2")
    print(f"Number of views: {gen.num_views}")

    y = gen(x)
    print(f"Output shape: {y.shape}")
    assert y.shape == (2, 3, 2560, 64), f"Expected (2, 3, 2560, 64), got {y.shape}"
    print("✓ Test passed!")


def test_splits_w():
    """Test splits along W dimension (dim=3)"""
    print("\n" + "="*60)
    print("Test 2: Splits along W (dim=3)")
    print("="*60)

    x = torch.randn(2, 3, 64, 2560)  # (N, C, H, W)
    print(f"Input shape: {x.shape}")

    gen = ViTUNetCrossView(
        features=128,
        n_heads=4,
        n_blocks=2,
        ffn_features=256,
        embed_features=64,
        activ='gelu',
        norm='layer',
        input_shape=(3, 64, 2560),
        output_shape=(3, 64, 2560),
        unet_features_list=[32, 64],
        unet_activ='relu',
        unet_norm='instance',
        split_sizes=[800, 800, 960],
        split_dim=3,  # W dimension
        activ_output='tanh'
    )

    print(f"Model created with split_sizes=[800, 800, 960], split_dim=3")
    print(f"Number of views: {gen.num_views}")

    y = gen(x)
    print(f"Output shape: {y.shape}")
    assert y.shape == (2, 3, 64, 2560), f"Expected (2, 3, 64, 2560), got {y.shape}"
    print("✓ Test passed!")


def test_equal_splits():
    """Test equal splits (backward-compatible behavior)"""
    print("\n" + "="*60)
    print("Test 3: Equal splits (backward-compatible)")
    print("="*60)

    x = torch.randn(2, 3, 192, 64)
    print(f"Input shape: {x.shape}")

    gen = ViTUNetCrossView(
        features=128,
        n_heads=4,
        n_blocks=2,
        ffn_features=256,
        embed_features=64,
        activ='gelu',
        norm='layer',
        input_shape=(3, 192, 64),
        output_shape=(3, 192, 64),
        unet_features_list=[32, 64],
        unet_activ='relu',
        unet_norm='instance',
        split_sizes=[64, 64, 64],  # equal splits
        split_dim=2,
        activ_output='tanh'
    )

    print(f"Model created with split_sizes=[64, 64, 64], split_dim=2")
    print(f"Number of views: {gen.num_views}")

    y = gen(x)
    print(f"Output shape: {y.shape}")
    assert y.shape == (2, 3, 192, 64), f"Expected (2, 3, 192, 64), got {y.shape}"
    print("✓ Test passed!")


def test_shape_validation():
    """Test shape validation (should fail)"""
    print("\n" + "="*60)
    print("Test 4: Shape validation (should fail)")
    print("="*60)

    print("Attempting to create model with mismatched split_sizes...")

    try:
        gen = ViTUNetCrossView(
            features=128,
            n_heads=4,
            n_blocks=2,
            ffn_features=256,
            embed_features=64,
            activ='gelu',
            norm='layer',
            input_shape=(3, 2048, 64),  # Mismatched!
            output_shape=(3, 2048, 64),
            unet_features_list=[32, 64],
            unet_activ='relu',
            unet_norm='instance',
            split_sizes=[800, 800, 960],  # sum = 2560, but input H = 2048
            split_dim=2,
            activ_output='tanh'
        )
        print("✗ Test failed: Expected ValueError but model was created successfully")
        return False
    except ValueError as e:
        print(f"✓ Test passed: Caught expected ValueError: {e}")
        return True


def test_two_views():
    """Test with 2 views instead of 3"""
    print("\n" + "="*60)
    print("Test 5: Two views with unequal splits")
    print("="*60)

    x = torch.randn(2, 3, 1000, 64)  # (N, C, H, W)
    print(f"Input shape: {x.shape}")

    gen = ViTUNetCrossView(
        features=128,
        n_heads=4,
        n_blocks=2,
        ffn_features=256,
        embed_features=64,
        activ='gelu',
        norm='layer',
        input_shape=(3, 1000, 64),
        output_shape=(3, 1000, 64),
        unet_features_list=[32, 64],
        unet_activ='relu',
        unet_norm='instance',
        split_sizes=[600, 400],  # Only 2 views
        split_dim=2,
        activ_output='tanh'
    )

    print(f"Model created with split_sizes=[600, 400], split_dim=2")
    print(f"Number of views: {gen.num_views}")

    y = gen(x)
    print(f"Output shape: {y.shape}")
    assert y.shape == (2, 3, 1000, 64), f"Expected (2, 3, 1000, 64), got {y.shape}"
    print("✓ Test passed!")


def test_four_views():
    """Test with 4 views"""
    print("\n" + "="*60)
    print("Test 6: Four views with mixed split sizes")
    print("="*60)

    x = torch.randn(1, 3, 3200, 64)  # (N, C, H, W)
    print(f"Input shape: {x.shape}")

    gen = ViTUNetCrossView(
        features=128,
        n_heads=4,
        n_blocks=2,
        ffn_features=256,
        embed_features=64,
        activ='gelu',
        norm='layer',
        input_shape=(3, 3200, 64),
        output_shape=(3, 3200, 64),
        unet_features_list=[32, 64],
        unet_activ='relu',
        unet_norm='instance',
        split_sizes=[800, 800, 800, 800],  # 4 equal views
        split_dim=2,
        activ_output='tanh'
    )

    print(f"Model created with split_sizes=[800, 800, 800, 800], split_dim=2")
    print(f"Number of views: {gen.num_views}")

    y = gen(x)
    print(f"Output shape: {y.shape}")
    assert y.shape == (1, 3, 3200, 64), f"Expected (1, 3, 3200, 64), got {y.shape}"
    print("✓ Test passed!")


if __name__ == '__main__':
    print("Testing new splitting policy functionality for ViTUNetCrossView")
    print("="*60)

    try:
        test_unequal_splits_h()
        test_splits_w()
        test_equal_splits()
        test_shape_validation()
        test_two_views()
        test_four_views()

        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
