---
generated: 2026-04-29
source-hash: 8f38aa219801ab66
---

# wirecell/pytorch

PyTorch integration for Wire-Cell Toolkit, providing torch script modules and a CLI for generating serialized torch artifacts. The primary focus is a scriptable DFT (Discrete Fourier Transform) module that wraps PyTorch's FFT routines to match the Wire-Cell toolkit's 1D, batched-1D, and 2D DFT API conventions.

## Modules

| Module | Purpose | Key Symbols |
|--------|---------|-------------|
| `script` | Torch script-compatible FFT wrapper module | `DFT` |
| `__main__` | CLI entry point for pytorch subcommands | `cli`, `make_dft` |

## CLI Commands

| Command | Description |
|---------|-------------|
| `make-dft` | Generate the DFT torch script module into given file. |

## Dependencies

| Dependency | Usage |
|-----------|-------|
| `wirecell.util.cli` | CLI context and logging utilities |
