#!/usr/bin/env python
"""
Example usage of the new ViTUNetCrossView with splitting policy.

This demonstrates how to use the single-image API with custom splits.
"""

import torch
from uvcgan2.models.generator.vitunet_crossview import ViTUNetCrossView


def example_unequal_splits():
    """Example: Process a single image with unequal splits along H"""
    print("="*60)
    print("Example: Unequal splits along H dimension")
    print("="*60)

    # Single input image with H=2560
    x = torch.randn(2, 3, 2560, 64)
    print(f"Input shape: {x.shape}")

    # Create model with unequal splits: [800, 800, 960]
    model = ViTUNetCrossView(
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
        split_sizes=[800, 800, 960],  # Unequal splits!
        split_dim=2,  # Split along H dimension
        activ_output='tanh'
    )

    # Forward pass: single image in, single image out
    y = model(x)
    print(f"Output shape: {y.shape}")
    print("✓ Successfully processed image with unequal splits!\n")


def example_two_views():
    """Example: Split into just 2 views"""
    print("="*60)
    print("Example: Two views with different sizes")
    print("="*60)

    x = torch.randn(1, 3, 1200, 64)
    print(f"Input shape: {x.shape}")

    model = ViTUNetCrossView(
        features=128,
        n_heads=4,
        n_blocks=2,
        ffn_features=256,
        embed_features=64,
        activ='gelu',
        norm='layer',
        input_shape=(3, 1200, 64),
        output_shape=(3, 1200, 64),
        unet_features_list=[32, 64],
        unet_activ='relu',
        unet_norm='instance',
        split_sizes=[700, 500],  # 2 unequal views
        split_dim=2,
        activ_output='tanh'
    )

    y = model(x)
    print(f"Output shape: {y.shape}")
    print("✓ Successfully processed with 2 views!\n")


def example_split_along_w():
    """Example: Split along W dimension instead of H"""
    print("="*60)
    print("Example: Split along W dimension")
    print("="*60)

    x = torch.randn(1, 3, 64, 2560)
    print(f"Input shape: {x.shape}")

    model = ViTUNetCrossView(
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
        split_dim=3,  # Split along W dimension!
        activ_output='tanh'
    )

    y = model(x)
    print(f"Output shape: {y.shape}")
    print("✓ Successfully processed with W-dimension splits!\n")


if __name__ == '__main__':
    print("\nViTUNetCrossView - New Single-Image API Examples")
    print("="*60)
    print()

    example_unequal_splits()
    example_two_views()
    example_split_along_w()

    print("="*60)
    print("All examples completed successfully!")
    print("="*60)
