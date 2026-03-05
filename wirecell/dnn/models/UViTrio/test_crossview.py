#!/usr/bin/env python3
"""
Test script for the ViTUNetCrossViewGenerator.
"""

import torch
from uvcgan2.models.generator.vitunet_crossview import ViTUNetCrossViewGenerator


def test_crossview_generator():
    """Test that the cross-view generator can be instantiated and run."""

    # Configuration
    batch_size = 2
    input_shape = (3, 64, 64)  # (C, H, W)
    output_shape = (3, 64, 64)

    # For now, test with identity bottleneck instead of transformer
    from uvcgan2.torch.layers.crossview import MultiViewUNet

    # Create MultiViewUNet directly with 3 views
    generator_net = MultiViewUNet(
        num_views=3,
        features_list=[32, 64, 128],
        activ='relu',
        norm='instance',
        image_shape=input_shape,
        downsample='conv',
        upsample='upsample-conv',
        rezero=False,
    )

    # Set an identity bottleneck for testing
    bottleneck_shape = generator_net.get_inner_shape()
    print(f"Bottleneck expects shape: {bottleneck_shape}")

    # Use identity function as bottleneck
    generator_net.set_bottleneck(torch.nn.Identity())

    print(f"✓ MultiViewUNet created successfully")
    print(f"  Parameters: {sum(p.numel() for p in generator_net.parameters()):,}")

    # Create dummy input (3 views)
    x0 = torch.randn(batch_size, *input_shape)
    x1 = torch.randn(batch_size, *input_shape)
    x2 = torch.randn(batch_size, *input_shape)

    print(f"✓ Input tensors created")
    print(f"  x0 shape: {x0.shape}")
    print(f"  x1 shape: {x1.shape}")
    print(f"  x2 shape: {x2.shape}")

    # Forward pass with list input
    with torch.no_grad():
        outputs = generator_net([x0, x1, x2])
        y0, y1, y2 = outputs

    print(f"✓ Forward pass successful")
    print(f"  y0 shape: {y0.shape}")
    print(f"  y1 shape: {y1.shape}")
    print(f"  y2 shape: {y2.shape}")

    # Verify output shapes
    assert y0.shape == (batch_size, *output_shape), f"y0 shape mismatch"
    assert y1.shape == (batch_size, *output_shape), f"y1 shape mismatch"
    assert y2.shape == (batch_size, *output_shape), f"y2 shape mismatch"

    print(f"✓ All output shapes correct")

    # Check that outputs are different (not identical copies)
    assert not torch.allclose(y0, y1), "y0 and y1 should be different"
    assert not torch.allclose(y1, y2), "y1 and y2 should be different"
    assert not torch.allclose(y0, y2), "y0 and y2 should be different"

    print(f"✓ Outputs are distinct (not identical)")

    print("\n" + "="*50)
    print("ALL TESTS PASSED!")
    print("="*50)
    print("\nNote: Tested with identity bottleneck. Transformer integration pending.")


def test_two_views():
    """Test MultiViewUNet with 2 views."""
    print("\n" + "="*50)
    print("Testing with 2 views")
    print("="*50)

    from uvcgan2.torch.layers.crossview import MultiViewUNet

    batch_size = 2
    input_shape = (3, 64, 64)

    # Create MultiViewUNet with 2 views
    generator_net = MultiViewUNet(
        num_views=2,
        features_list=[32, 64, 128],
        activ='relu',
        norm='instance',
        image_shape=input_shape,
        downsample='conv',
        upsample='upsample-conv',
        rezero=False,
    )

    generator_net.set_bottleneck(torch.nn.Identity())

    print(f"✓ MultiViewUNet (2 views) created successfully")
    print(f"  Bottleneck shape: {generator_net.get_inner_shape()}")

    # Create inputs
    x0 = torch.randn(batch_size, *input_shape)
    x1 = torch.randn(batch_size, *input_shape)

    # Test with list input
    with torch.no_grad():
        outputs = generator_net([x0, x1])
        assert len(outputs) == 2, f"Expected 2 outputs, got {len(outputs)}"
        y0, y1 = outputs

    print(f"✓ Forward pass with list input successful")
    print(f"  y0 shape: {y0.shape}, y1 shape: {y1.shape}")

    assert y0.shape == (batch_size, *input_shape), "y0 shape mismatch"
    assert y1.shape == (batch_size, *input_shape), "y1 shape mismatch"
    assert not torch.allclose(y0, y1), "Outputs should be different"

    # Test with tuple input
    with torch.no_grad():
        outputs = generator_net((x0, x1))
        assert isinstance(outputs, tuple), "Expected tuple output for tuple input"
        assert len(outputs) == 2, f"Expected 2 outputs, got {len(outputs)}"

    print(f"✓ Forward pass with tuple input successful")
    print(f"✓ All 2-view tests passed!")


def test_four_views():
    """Test MultiViewUNet with 4 views."""
    print("\n" + "="*50)
    print("Testing with 4 views")
    print("="*50)

    from uvcgan2.torch.layers.crossview import MultiViewUNet

    batch_size = 2
    input_shape = (3, 64, 64)

    # Create MultiViewUNet with 4 views
    generator_net = MultiViewUNet(
        num_views=4,
        features_list=[32, 64, 128],
        activ='relu',
        norm='instance',
        image_shape=input_shape,
        downsample='conv',
        upsample='upsample-conv',
        rezero=False,
    )

    generator_net.set_bottleneck(torch.nn.Identity())

    print(f"✓ MultiViewUNet (4 views) created successfully")
    print(f"  Bottleneck shape: {generator_net.get_inner_shape()}")

    # Create inputs
    inputs = [torch.randn(batch_size, *input_shape) for _ in range(4)]

    # Test with list input
    with torch.no_grad():
        outputs = generator_net(inputs)
        assert len(outputs) == 4, f"Expected 4 outputs, got {len(outputs)}"

    print(f"✓ Forward pass with list input successful")
    for i, out in enumerate(outputs):
        print(f"  y{i} shape: {out.shape}")
        assert out.shape == (batch_size, *input_shape), f"y{i} shape mismatch"

    # Verify outputs are distinct
    for i in range(4):
        for j in range(i+1, 4):
            assert not torch.allclose(outputs[i], outputs[j]), \
                f"Outputs {i} and {j} should be different"

    print(f"✓ All outputs are distinct")
    print(f"✓ All 4-view tests passed!")


if __name__ == "__main__":
    test_crossview_generator()
    test_two_views()
    test_four_views()
