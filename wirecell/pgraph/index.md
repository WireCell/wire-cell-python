---
generated: 2026-04-29
source-hash: 8fe89b29730b57b2
---

# wirecell/pgraph

Provides tools for visualizing Wire Cell Toolkit processing graph configurations as GraphViz diagrams. Converts WCT JSON/Jsonnet configuration files — which describe directed dataflow graphs of typed nodes connected by edges — into `.dot` files or rendered image formats. Supports display of node parameters, service dependencies, and configurable layout options.

## Modules

| Module | Purpose | Key Symbols |
|--------|---------|-------------|
| `__main__` | CLI entry point for pgraph tools | `cli`, `Node`, `dotify`, `resolve_path`, `uses_to_params` |

## CLI Commands

| Command | Description |
|---------|-------------|
| `dotify` | Convert a WCT cfg to a GraphViz dot or rendered file. |

## Dependencies

| Import | Purpose |
|--------|---------|
| `wirecell.units` | Physical unit constants |
| `wirecell.util.jsio` | JSON/Jsonnet file I/O |
| `wirecell.util.cli.jsonnet_loader` | Decorator for loading Jsonnet/JSON input files |
| `wirecell.util.cli.context` | CLI context group helper |
| `wirecell.util.cli.log` | Shared logger |
