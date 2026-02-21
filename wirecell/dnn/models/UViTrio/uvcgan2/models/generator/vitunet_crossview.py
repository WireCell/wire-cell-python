# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes

import torch
from torch import nn

from uvcgan2.torch.layers.transformer import PixelwiseViT
from uvcgan2.torch.layers.crossview import MultiViewUNet, SplitAwareBottleneck
from uvcgan2.torch.select import get_activ_layer


class ViTUNetCrossView(nn.Module):
    """
    A ViTUNet that processes a single input image by splitting it at user-defined
    intervals along an arbitrary dimension. The splits are processed through separate
    UNet encoder paths, concatenated at the transformer bottleneck, decoded separately,
    and recombined into a single output image.

    This is useful for handling variable-sized regions within a single large image,
    where different regions may have different characteristics or importance.
    """

    def __init__(
        self, features, n_heads, n_blocks, ffn_features, embed_features,
        activ, norm, input_shape, output_shape,
        unet_features_list, unet_activ, unet_norm,
        split_sizes, split_dim,
        unet_downsample='conv',
        unet_upsample='upsample-conv',
        unet_rezero=False,
        rezero=True,
        activ_output=None,
        **kwargs
    ):
        """
        Args:
            features: Number of features for transformer
            n_heads: Number of attention heads
            n_blocks: Number of transformer blocks
            ffn_features: Feed-forward network features
            embed_features: Embedding features
            activ: Activation function for transformer
            norm: Normalization for transformer
            input_shape: Input image shape (C, H, W)
            output_shape: Output image shape (C, H, W)
            unet_features_list: List of feature dimensions for UNet levels
            unet_activ: Activation function for UNet
            unet_norm: Normalization for UNet
            split_sizes: List of sizes for splitting input image (e.g., [800, 800, 960])
            split_dim: Dimension to split along (2 for H, 3 for W)
            unet_downsample: Downsampling method
            unet_upsample: Upsampling method
            unet_rezero: Whether to use rezero in UNet decoder blocks
            rezero: Whether to use rezero in transformer
            activ_output: Output activation function

        Note: The sum of split_sizes must equal the size of input_shape along split_dim.
        """
        super().__init__(**kwargs)

        assert input_shape == output_shape

        self.image_shape = input_shape
        self.split_sizes = split_sizes
        self.split_dim = split_dim
        self.num_views = len(split_sizes)

        # Validate that split_sizes sum matches input_shape
        img_shape_idx = split_dim - 1  # Convert from torch dim (0=N,1=C,2=H,3=W) to shape idx
        if sum(split_sizes) != input_shape[img_shape_idx]:
            raise ValueError(
                f"Sum of split_sizes ({sum(split_sizes)}) must equal "
                f"input_shape[{img_shape_idx}] ({input_shape[img_shape_idx]}) "
                f"for split_dim={split_dim}"
            )

        # Create the multi-view UNet structure with num_views from split_sizes
        self.net = MultiViewUNet(
            self.num_views, unet_features_list, unet_activ, unet_norm, input_shape,
            unet_downsample, unet_upsample, unet_rezero,
            split_sizes=split_sizes, split_dim=split_dim
        )

        # Get the actual bottleneck shape (after all encoding)
        bottleneck_shape = self.net.get_inner_shape()

        # Create the transformer bottleneck
        # It operates on the concatenated feature maps
        transformer = PixelwiseViT(
            features, n_heads, n_blocks, ffn_features, embed_features,
            activ, norm,
            image_shape=bottleneck_shape,
            rezero=rezero
        )

        # Compute the encoded split sizes at the bottleneck level
        # by encoding sample tensors (done in get_inner_shape)
        with torch.no_grad():
            # Create sample views
            if split_dim == 2:  # H dimension
                sample_views = [
                    torch.zeros(1, input_shape[0], size, input_shape[2])
                    for size in split_sizes
                ]
            elif split_dim == 3:  # W dimension
                sample_views = [
                    torch.zeros(1, input_shape[0], input_shape[1], size)
                    for size in split_sizes
                ]
            else:
                raise ValueError(f"Unsupported split_dim: {split_dim}")

            # Encode through the network to get bottleneck split sizes
            y_list = [self.net.input_layers[i](sample_views[i]) for i in range(self.num_views)]
            curr_list = y_list
            block = self.net.unet
            while hasattr(block, 'encoders'):
                encoded = [block.encoders[i](curr_list[i]) for i in range(self.num_views)]
                curr_list = [y for y, r in encoded]
                if block.inner_module is None or not hasattr(block.inner_module, 'encoders'):
                    break
                block = block.inner_module

            # Get the encoded split sizes
            if split_dim == 2:
                encoded_split_sizes = [y.shape[2] for y in curr_list]
            elif split_dim == 3:
                encoded_split_sizes = [y.shape[3] for y in curr_list]
            else:
                raise ValueError(f"Unsupported split_dim: {split_dim}")

        # Wrap in SplitAwareBottleneck with ENCODED split sizes
        bottleneck = SplitAwareBottleneck(transformer, encoded_split_sizes, split_dim)
        self.net.set_bottleneck(bottleneck)

        # Output activation
        self.output = get_activ_layer(activ_output)

    def forward(self, x):
        """
        Forward pass with single input image.

        Args:
            x: Input image (N, C, H, W) where one dimension equals sum(split_sizes)

        Returns:
            Output image (N, C, H, W) - same shape as input
        """
        # Split input into views
        views = list(torch.split(x, self.split_sizes, dim=self.split_dim))

        # Process through MultiViewUNet
        outputs = self.net(views)  # returns list/tuple

        # Concatenate results along same dimension
        y = torch.cat(outputs, dim=self.split_dim)

        # Apply output activation
        y = self.output(y)

        return y
