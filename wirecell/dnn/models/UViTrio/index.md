---
generated: 2026-04-29
source-hash: a21f62e8a59d5b53
children-hash: a250b2a6566272cb
---

# wirecell/dnn/models/UViTrio

Development and testing workspace for the `ViTUNetCrossView` model — a Vision Transformer bottlenecked U-Net that processes large images by splitting them into multiple views along spatial dimensions. Provides example scripts, profiling utilities, and test suites for validating the splitting policy that enables unequal and arbitrary-size view decomposition.

## Modules

| Module | Purpose | Key Symbols |
|---|---|---|
| `uvcgan2` | Core framework: generator architectures and PyTorch utilities | `ViTUNetGenerator`, `ViTUNetCrossView`, `MultiViewUNet` |
| `example_split_usage` | Runnable examples of single-image API with custom split configurations | `example_unequal_splits`, `example_two_views`, `example_split_along_w` |
| `profile_split_model` | Torch profiler harness for `ViTUNetCrossView` with throughput benchmarking | `create_model`, `profile_model`, `benchmark_throughput` |
| `test_crossview` | Tests for `MultiViewUNet` with 2-, 3-, and 4-view configurations using identity bottleneck | `test_crossview_generator`, `test_two_views`, `test_four_views` |
| `test_split_manual` | Manual tensor-shape trace to inspect bottleneck dimensions after downsampling | `test_manual_computation` |
| `test_split_policy` | Full integration tests for `ViTUNetCrossView` splitting policy including shape validation | `test_unequal_splits_h`, `test_splits_w`, `test_shape_validation` |
| `test_split_policy_simple` | Simplified splitting tests using identity bottleneck to isolate split logic from transformer | `test_unequal_splits_h`, `test_splits_w`, `test_equal_splits` |

## Dependencies

- `uvcgan2.models.generator.vitunet_crossview` — `ViTUNetCrossView`, `ViTUNetCrossViewGenerator`
- `uvcgan2.torch.layers.crossview` — `MultiViewUNet`, `SplitAwareBottleneck`
- `torch`, `torch.profiler` — tensor ops, `profile`, `ProfilerActivity`
