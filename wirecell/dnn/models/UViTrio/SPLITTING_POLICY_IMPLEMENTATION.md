# Splitting Policy Implementation for ViTUNetCrossView

## Summary

Implemented a flexible splitting policy that enables `ViTUNetCrossView` to accept a single input image and split it at user-defined intervals along an arbitrary dimension for processing.

## Key Changes

### 1. New `SplitAwareBottleneck` Class
**File**: `uvcgan2/torch/layers/crossview.py`

- Wrapper around bottleneck modules (e.g., PixelwiseViT) that stores splitting policy information
- Validates that transformer shape matches the sum of split sizes
- Exposes `get_split_info()` method to communicate split policy to parent blocks

```python
class SplitAwareBottleneck(nn.Module):
    def __init__(self, transformer, split_sizes, split_dim):
        # split_sizes: e.g., [200, 200, 240] (encoded sizes at bottleneck)
        # split_dim: 2 for H, 3 for W
```

### 2. Modified `MultiViewUNetBlock`
**File**: `uvcgan2/torch/layers/crossview.py`

- Enhanced `forward()` to support arbitrary split dimensions
- Queries bottleneck for split policy via `get_split_info()`
- Uses custom split sizes or falls back to equal chunks
- Concatenates and splits along the dimension specified by bottleneck

**Key changes**:
- Line 150-166: Dynamic split dimension and custom split sizes
- Falls back to equal splits for backward compatibility

### 3. Modified `MultiViewUNet`
**File**: `uvcgan2/torch/layers/crossview.py`

- Added `split_sizes` and `split_dim` parameters to constructor
- Implemented smart `get_inner_shape()` that computes actual encoded sizes by encoding sample tensors
- Accounts for different downsampling factors for unequal splits

**Key feature**:
- `get_inner_shape()` now actually encodes sample tensors to determine the correct concatenated shape at the bottleneck

### 4. Rewritten `ViTUNetCrossView` API
**File**: `uvcgan2/models/generator/vitunet_crossview.py`

**Breaking Change**: New single-image API

**Old API**:
```python
y0, y1, y2 = model(x0, x1, x2)  # 3 separate images
```

**New API**:
```python
y = model(x)  # Single image, split internally
```

**New Required Parameters**:
- `split_sizes`: List of sizes for splitting (e.g., `[800, 800, 960]`)
- `split_dim`: Dimension to split along (2=H, 3=W)

**Constructor enhancements**:
- Validates that sum of `split_sizes` matches `input_shape` dimension
- Computes encoded split sizes at bottleneck by encoding sample tensors
- Creates `SplitAwareBottleneck` with correct encoded sizes
- Passes split policy to `MultiViewUNet`

**Forward method**:
- Splits input into views using `torch.split(x, split_sizes, dim=split_dim)`
- Processes through `MultiViewUNet`
- Concatenates results along same dimension
- Returns single output image

## Usage Examples

### Unequal Splits Along H
```python
x = torch.randn(2, 3, 2560, 64)  # Single image
model = ViTUNetCrossView(
    ...,
    input_shape=(3, 2560, 64),
    split_sizes=[800, 800, 960],  # Unequal splits!
    split_dim=2  # H dimension
)
y = model(x)  # Returns (2, 3, 2560, 64)
```

### Split Along W Dimension
```python
x = torch.randn(2, 3, 64, 2560)
model = ViTUNetCrossView(
    ...,
    input_shape=(3, 64, 2560),
    split_sizes=[800, 800, 960],
    split_dim=3  # W dimension
)
y = model(x)  # Returns (2, 3, 64, 2560)
```

### Two Views (Not Limited to 3)
```python
x = torch.randn(1, 3, 1200, 64)
model = ViTUNetCrossView(
    ...,
    input_shape=(3, 1200, 64),
    split_sizes=[700, 500],  # Just 2 views
    split_dim=2
)
y = model(x)
```

### Equal Splits (Backward-Compatible Behavior)
```python
x = torch.randn(2, 3, 192, 64)
model = ViTUNetCrossView(
    ...,
    input_shape=(3, 192, 64),
    split_sizes=[64, 64, 64],  # Equal splits
    split_dim=2
)
y = model(x)
```

## Architecture Flow

1. **Input**: Single image `(N, C, H, W)` where H or W = sum(split_sizes)

2. **Split**: `torch.split(x, split_sizes, dim=split_dim)` → list of views

3. **Encode**: Each view processed through separate encoder paths
   - View 1: 800 → (downsampling) → 200
   - View 2: 800 → (downsampling) → 200
   - View 3: 960 → (downsampling) → 240

4. **Concatenate**: At bottleneck, concat along same dimension
   - Result: (N, C, 640, W) where 640 = 200+200+240

5. **Transform**: PixelwiseViT processes concatenated features

6. **Split**: Back into separate views using encoded split sizes `[200, 200, 240]`

7. **Decode**: Each view decoded through separate decoder paths

8. **Recombine**: Concatenate decoded views → `(N, C, 2560, W)`

## Design Rationale

### Why Wrap the Bottleneck?
The splitting policy is implemented as a property of the bottleneck because:
1. **Logical coupling**: Split sizes must match transformer's expected shape
2. **Separation of concerns**: UNet blocks query bottleneck rather than managing policy themselves
3. **Extensibility**: Other bottleneck types can have different policies
4. **Validation**: Bottleneck can validate shape compatibility at construction time

### Why Compute Encoded Sizes?
Input split sizes (e.g., `[800, 800, 960]`) differ from encoded sizes at bottleneck (e.g., `[200, 200, 240]`) due to downsampling. The implementation:
- Takes INPUT split sizes from user (intuitive)
- Automatically computes ENCODED split sizes (by encoding sample tensors)
- Passes encoded sizes to SplitAwareBottleneck (correct for transformer)

This makes the API intuitive (users specify input dimensions) while maintaining correctness internally.

## Testing

### Test Files
- `test_split_policy_simple.py`: Tests with identity bottleneck
- `test_split_manual.py`: Manual computation to verify encoding sizes
- `example_split_usage.py`: User-friendly examples

### Test Coverage
- ✓ Unequal splits along H dimension
- ✓ Unequal splits along W dimension
- ✓ Equal splits (backward-compatible)
- ✓ Variable number of views (2, 3, 4, etc.)
- ✓ Shape validation (rejects mismatched sizes)

All tests pass successfully!

## Backward Compatibility

**Breaking Change**: The forward signature has changed from `forward(x0, x1, x2)` to `forward(x)`.

**Migration Path**:
Users with separate images must concatenate them first:
```python
# Old code:
# y0, y1, y2 = model(x0, x1, x2)

# New code:
x = torch.cat([x0, x1, x2], dim=2)  # Concatenate along H
y = model(x)  # Returns concatenated output
# Split if needed: y0, y1, y2 = torch.split(y, [h0, h1, h2], dim=2)
```

## Files Modified

1. `UViTrio/uvcgan2/torch/layers/crossview.py`
   - Added `SplitAwareBottleneck` class (~50 lines)
   - Modified `MultiViewUNetBlock.forward()` (~20 lines changed)
   - Modified `MultiViewUNet.__init__()` and `get_inner_shape()` (~50 lines changed)

2. `UViTrio/uvcgan2/models/generator/vitunet_crossview.py`
   - Rewritten constructor (~40 lines changed)
   - Rewritten `forward()` method (~20 lines changed)
   - Updated imports

**Total**: ~180 lines changed/added across 2 files

## Performance Considerations

- `get_inner_shape()` encodes sample tensors once during initialization
- This adds minimal overhead (~few ms) at model construction time
- No runtime performance impact - splitting/concatenation are native PyTorch ops
- Memory usage unchanged (same total tensor sizes)

## Future Enhancements

Potential extensions (not implemented):
1. Support for overlapping splits
2. Dynamic split sizes per forward pass
3. Non-contiguous splits (e.g., [0:800, 1000:1800, 2000:2960])
4. Per-view processing weights/attention
