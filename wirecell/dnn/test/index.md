---
generated: 2026-04-29
source-hash: 95c52c1e59533620
---

# wirecell/dnn/test

Test and utility scripts for the `wirecell.dnn` package. Contains pytest-based integration tests for the DNNROI data pipeline and a standalone SVG logo generator. Tests cover dataset loading, timing benchmarks, and parametrized lazy/cache/grad configurations.

## Modules

| Module | Purpose | Key Symbols |
|--------|---------|-------------|
| `test_dnnroi.py` | Integration tests for DNNROI data loading and timing | `test_data`, `test_data_timing`, `recps`, `trups` |
| `gen-dnnroi-logo.py` | Standalone script generating an SVG logo for DNNROI using randomized clip-path rectangles | `generate_random_rectangle`, `generate_svg_content` |

## Dependencies

| Import | Role |
|--------|------|
| `wirecell.dnn.data.common.DatasetParams` | Configures lazy/cache/grad dataset loading behavior |
| `wirecell.dnn.apps.dnnroi` | Provides `dnnroi.data.Rec` and `dnnroi.data.Tru` dataset classes under test |
