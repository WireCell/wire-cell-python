# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes

import torch
from torch import nn

from .unet import UNetEncBlock, UNetDecBlock


class SplitAwareBottleneck(nn.Module):
    """
    Wrapper around a bottleneck module (e.g., PixelwiseViT) that stores
    splitting policy information for custom splits.

    This allows the bottleneck to communicate how concatenated features
    should be split back into separate views.
    """

    def __init__(self, transformer, split_sizes, split_dim):
        """
        Args:
            transformer: The bottleneck module (e.g., PixelwiseViT)
            split_sizes: List of sizes for each split (e.g., [800, 800, 960])
            split_dim: Dimension to split/concat (2=H, 3=W, etc.)
        """
        super().__init__()
        self.transformer = transformer
        self.split_sizes = split_sizes
        self.split_dim = split_dim

        # Validate that transformer expects concatenated shape
        self._validate_shapes()

    def _validate_shapes(self):
        """Check that transformer image_shape matches sum of split_sizes"""
        # split_dim uses torch indexing (0=N, 1=C, 2=H, 3=W)
        # transformer.image_shape uses (C, H, W) indexing
        img_shape_idx = self.split_dim - 1  # convert to image_shape indexing
        expected_total = sum(self.split_sizes)
        actual_total = self.transformer.image_shape[img_shape_idx]

        if expected_total != actual_total:
            raise ValueError(
                f"Split sizes sum to {expected_total} but transformer expects "
                f"{actual_total} along dimension {self.split_dim}"
            )

    def forward(self, x):
        """Pass through to wrapped transformer"""
        return self.transformer(x)

    def get_split_info(self):
        """Return (split_sizes, split_dim) for custom splitting"""
        return (self.split_sizes, self.split_dim)


class MultiViewUNetBlock(nn.Module):
    """
    A UNet block that processes multiple images in parallel through separate
    encoder paths, optionally concatenates them at the inner module level,
    and decodes them separately.

    This maintains the nested structure of UNetBlock but handles multiple
    parallel streams.
    """

    def __init__(
        self, num_views, features, activ, norm, image_shape, downsample, upsample,
        rezero=True, **kwargs
    ):
        """
        Args:
            num_views: Number of parallel views to process
            features: Number of features for the conv blocks
            activ: Activation function
            norm: Normalization layer type
            image_shape: Shape of input (C, H, W)
            downsample: Downsampling method
            upsample: Upsampling method
            rezero: Whether to use rezero in decoder blocks

        Note: This always concatenates along the height (H) dimension.
        """
        super().__init__(**kwargs)

        self.num_views = num_views

        # Create parallel encoder blocks using ModuleList
        self.encoders = nn.ModuleList([
            UNetEncBlock(features, activ, norm, downsample, image_shape)
            for _ in range(num_views)
        ])

        # All encoders produce the same output shape
        self.inner_shape = self.encoders[0].output_shape
        self.inner_module = None

        # Create parallel decoder blocks using ModuleList
        self.decoders = nn.ModuleList([
            UNetDecBlock(
                self.inner_shape, image_shape[0], self.inner_shape[0],
                activ, norm, upsample, rezero
            )
            for _ in range(num_views)
        ])

    def get_inner_shape(self):
        """
        Returns the shape after concatenation along height.
        Returns: (C, num_views*H, W)
        """
        (C, H, W) = self.inner_shape
        return (C, self.num_views * H, W)

    def set_inner_module(self, module):
        """Set the inner module (could be another MultiViewUNetBlock or transformer)"""
        self.inner_module = module

    def get_inner_module(self):
        """Get the inner module"""
        return self.inner_module

    def forward(self, images):
        """
        Forward pass with multiple input images.

        Args:
            images: List or tuple of tensors, each of shape (N, C, H, W)

        Returns:
            List or tuple of output tensors (matching input type), each of shape (N, C, H, W)
        """
        # Preserve input type (list vs tuple)
        is_tuple = isinstance(images, tuple)

        # Encode each image with its corresponding encoder
        # Each encoder returns (y, r) where y is downsampled, r is skip connection
        encoded = [self.encoders[i](img) for i, img in enumerate(images)]
        y_list = [y for y, r in encoded]
        skip_list = [r for y, r in encoded]

        # Handle inner module
        if isinstance(self.inner_module, MultiViewUNetBlock):
            # Keep streams separate for nested MultiViewUNetBlocks
            y_list = self.inner_module(y_list if not is_tuple else tuple(y_list))
            # Convert back to list for processing
            if isinstance(y_list, tuple):
                y_list = list(y_list)
        else:
            # Get split info from bottleneck if available
            if hasattr(self.inner_module, 'get_split_info'):
                split_sizes, split_dim = self.inner_module.get_split_info()
            else:
                # Default: equal chunks along height (dimension 2)
                split_sizes = None
                split_dim = 2

            # Concatenate along specified dimension
            y_concat = torch.cat(y_list, dim=split_dim)
            y_concat = self.inner_module(y_concat)

            # Split back using custom sizes or equal chunks
            if split_sizes is not None:
                y_list = list(torch.split(y_concat, split_sizes, dim=split_dim))
            else:
                y_list = list(torch.chunk(y_concat, self.num_views, dim=split_dim))

        # Decode each stream separately with its skip connection
        outputs = [self.decoders[i](y_list[i], skip_list[i])
                   for i in range(self.num_views)]

        # Return same type as input
        return tuple(outputs) if is_tuple else outputs


class MultiViewUNet(nn.Module):
    """
    A UNet that processes multiple images in parallel, concatenates them at
    the bottleneck, and decodes them separately.
    """

    def __init__(
        self, num_views, features_list, activ, norm, image_shape, downsample, upsample,
        rezero=True, split_sizes=None, split_dim=2, **kwargs
    ):
        """
        Args:
            num_views: Number of parallel views to process
            features_list: List of feature dimensions for each UNet level
            activ: Activation function
            norm: Normalization layer type
            image_shape: Input image shape (C, H, W) - full concatenated size
            downsample: Downsampling method
            upsample: Upsampling method
            rezero: Whether to use rezero in decoder blocks
            split_sizes: Optional list of split sizes for unequal splits
            split_dim: Dimension along which splitting occurs (2=H, 3=W)

        Note: If split_sizes is None, assumes equal splits.
        """
        super().__init__(**kwargs)

        self.num_views = num_views
        self.features_list = features_list
        self.image_shape = image_shape
        self.split_sizes = split_sizes
        self.split_dim = split_dim

        # Create input layers for each stream
        self._construct_input_layers(activ)
        self._construct_output_layers()

        # Build nested MultiViewUNetBlocks
        blocks = []
        curr_image_shape = (features_list[0], *image_shape[1:])

        for features in features_list:
            layer = MultiViewUNetBlock(
                num_views, features, activ, norm, curr_image_shape, downsample, upsample,
                rezero
            )
            curr_image_shape = layer.get_inner_shape()
            blocks.append(layer)

        # Link them together: each layer's inner_module is the next layer
        for idx in range(len(blocks) - 1):
            blocks[idx].set_inner_module(blocks[idx + 1])

        self.unet = blocks[0]

    def _construct_input_layers(self, activ):
        """Create separate input conv layers for each stream"""
        from uvcgan2.torch.select import get_activ_layer

        self.input_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    self.image_shape[0], self.features_list[0],
                    kernel_size=3, padding=1
                ),
                get_activ_layer(activ),
            )
            for _ in range(self.num_views)
        ])

    def _construct_output_layers(self):
        """Create separate output conv layers for each stream"""
        self.output_layers = nn.ModuleList([
            nn.Conv2d(
                self.features_list[0], self.image_shape[0], kernel_size=1
            )
            for _ in range(self.num_views)
        ])

    def get_innermost_block(self):
        """Navigate to the innermost MultiViewUNetBlock"""
        result = self.unet

        for _ in range(len(self.features_list) - 1):
            result = result.get_inner_module()

        return result

    def set_bottleneck(self, module):
        """
        Set the bottleneck module (e.g., transformer).
        This module should accept concatenated input.
        """
        self.get_innermost_block().set_inner_module(module)

    def get_bottleneck(self):
        """Get the bottleneck module"""
        return self.get_innermost_block().get_inner_module()

    def get_inner_shape(self):
        """
        Get the shape at the bottleneck (after concatenation).

        If split_sizes is provided, computes the actual concatenated size
        accounting for unequal splits by actually encoding sample tensors.
        """
        innermost = self.get_innermost_block()

        if self.split_sizes is None:
            # Default behavior: assume equal splits
            return innermost.get_inner_shape()

        # Compute actual encoded size by encoding sample tensors
        # This is the most reliable way since it accounts for actual downsampling
        with torch.no_grad():
            # Create sample views with actual split sizes
            if self.split_dim == 2:  # H dimension
                sample_views = [
                    torch.zeros(1, self.image_shape[0], size, self.image_shape[2])
                    for size in self.split_sizes
                ]
            elif self.split_dim == 3:  # W dimension
                sample_views = [
                    torch.zeros(1, self.image_shape[0], self.image_shape[1], size)
                    for size in self.split_sizes
                ]
            else:
                raise ValueError(f"Unsupported split_dim: {self.split_dim}")

            # Pass through input layers
            y_list = [self.input_layers[i](sample_views[i]) for i in range(self.num_views)]

            # Encode through all levels to get to bottleneck
            curr_list = y_list
            block = self.unet
            while hasattr(block, 'encoders'):
                encoded = [block.encoders[i](curr_list[i]) for i in range(self.num_views)]
                curr_list = [y for y, r in encoded]
                if block.inner_module is None or not hasattr(block.inner_module, 'encoders'):
                    break
                block = block.inner_module

            # Get the shape from the encoded tensors
            C = curr_list[0].shape[1]  # Channels

            if self.split_dim == 2:  # H dimension
                encoded_h_sizes = [y.shape[2] for y in curr_list]
                total_h = sum(encoded_h_sizes)
                W = curr_list[0].shape[3]
                return (C, total_h, W)
            elif self.split_dim == 3:  # W dimension
                H = curr_list[0].shape[2]
                encoded_w_sizes = [y.shape[3] for y in curr_list]
                total_w = sum(encoded_w_sizes)
                return (C, H, total_w)
            else:
                raise ValueError(f"Unsupported split_dim: {self.split_dim}")

    def forward(self, images):
        """
        Forward pass with multiple input images.

        Args:
            images: List or tuple of tensors, each of shape (N, C, H, W)

        Returns:
            List or tuple of output tensors (matching input type), each of shape (N, C, H, W)
        """
        # Preserve input type (list vs tuple)
        is_tuple = isinstance(images, tuple)

        # Pass through input layers
        y_list = [self.input_layers[i](images[i]) for i in range(self.num_views)]

        # Pass through nested UNet blocks
        y_list = self.unet(y_list if not is_tuple else tuple(y_list))
        # Convert back to list for processing
        if isinstance(y_list, tuple):
            y_list = list(y_list)

        # Pass through output layers
        outputs = [self.output_layers[i](y_list[i]) for i in range(self.num_views)]

        # Return same type as input
        return tuple(outputs) if is_tuple else outputs
