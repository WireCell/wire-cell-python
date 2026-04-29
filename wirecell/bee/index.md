---
generated: 2026-04-29
source-hash: db0478e1ac0f0d02
---

# wirecell/bee

Python package for working with Bee visualization server files and data. Provides structured loading of Bee's JSON/ZIP file formats, hierarchical summarization of reconstructed 3D cluster data, and CLI tools for inspection and comparison.

## Modules

| Module | Purpose | Key Symbols |
|--------|---------|-------------|
| `data` | Load and structure Bee JSON/ZIP files into typed objects | `Cluster`, `Grouping`, `Series`, `Ensemble`, `load`, `load_json`, `load_zip` |
| `ana` | Analyze and summarize Bee data structures at configurable depth levels | `Summary`, `levels`, `level_index` |
| `__main__` | CLI entry point for bee commands | `cmd_summary`, `cmd_diff` |

## CLI Commands

| Command | Description |
|---------|-------------|
| `summary` | Print summary of Bee files (.zip or .json) at a configurable depth level (point, shape, cluster, grouping, ensemble) |
| `diff` | Diff two Bee files by comparing their text summaries at a configurable depth level |

## Dependencies

| Import | Purpose |
|--------|---------|
| `wirecell.util.cli` | CLI context decorator |
| `wirecell.util.ario` | Archive I/O for reading ZIP file entries |
| `wirecell.util.points` | PCA computation (`pca_eigen`) for cluster shape analysis |
