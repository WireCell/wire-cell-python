---
generated: 2026-04-29
source-hash: 43d55e56153efdad
---

# wirecell/ls4gan

Tools for interfacing between LS4GAN (a generative adversarial network for liquid scintillator detector simulation) and the Wire Cell Toolkit. Provides CLI commands for converting GAN-produced NPZ arrays into WCT-compatible frame files and for computing comparison metrics between arrays.

## Modules

| Module | Purpose | Key Symbols |
|--------|---------|-------------|
| `__main__` | CLI entry point for ls4gan subcommands | `cli`, `npz_to_wct`, `comp_metric` |

## CLI Commands

| Command | Description |
|---------|-------------|
| `npz-to-wct` | Convert a 3D NPZ frame array (channel × tick × plane) to a WCT-compatible NPZ frame file, with optional transpose, channel remapping, linear scaling, and type casting |
| `comp-metric` | Compute L1 or L2 distance metrics between two NPZ files (2D WCT or 3D LS4GAN format) per plane (U, V, W) |

## Dependencies

| Import | Role |
|--------|------|
| `wirecell.units` | Physical unit constants |
| `wirecell.util.functions.unitify` | Parse unit-bearing string values |
| `wirecell.util.cli.context` | CLI context decorator for subcommand groups |
| `wirecell.util.ario` | Array I/O abstraction for loading frame data (used in `comp-metric`) |
