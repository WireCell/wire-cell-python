---
generated: 2026-04-29
source-hash: f374d2a18b93be94
---

# wirecell/aux/

Tools for validating and benchmarking the Wire-Cell Toolkit's auxiliary DFT (Discrete Fourier Transform) implementations against NumPy references. The package provides array generation, DFT operator wrappers, benchmark data handling, and system information collection to support correctness and performance testing of WCT's IDFT components.

## Modules

| Module | Purpose | Key Symbols |
|--------|---------|-------------|
| `idft` | DFT array generation, config creation, result comparison, and benchmark plotting | `gen_arrays`, `gen_config`, `get_arrays`, `save_arrays`, `fwd1d`, `inv1d`, `fwd2d`, `inv2d`, `fwd1b0`, `fwd1b1`, `plot_time`, `plot_plan_time` |
| `sysinfo` | Collect CPU and GPU hardware info for annotating benchmark results | `cpu`, `gpu`, `asdict` |

## CLI Commands

| Command | Description |
|---------|-------------|
| `run-idft` | Perform DFT transforms with `check_idft` and NumPy and compare results element-wise |
| `run-idft-bench` | Run `check_idft_bench`, augmenting JSON output with host system info |
| `plot-idft-bench` | Make PDF plots from one or more `check_idft_bench` output JSON files |

## Dependencies

| Import | Role |
|--------|------|
| `wirecell.util.jsio` | Load JSON/Jsonnet config files describing DFT operations |
| `wirecell.util.ario` | Read array archives (tar/bz2/gz) for input and output arrays |
| `wirecell.util.cli` | CLI context and logging utilities (`context`, `log`) |
