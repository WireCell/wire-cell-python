---
generated: 2026-04-29
source-hash: 567cab36e0110efa
---

# wirecell/gen/plots

Visualization utilities for wire-cell simulation generator output. Provides plotting routines for "morse" deposition patterns (transverse and longitudinal peak-width analysis) and legacy NumpySaver frame files, with the historic `numpy_saver` name preserved for backward compatibility.

## Modules

| Module | Purpose | Key Symbols |
|--------|---------|-------------|
| `morse` | Plots and Gaussian fitting of peak widths from the morse deposition pattern across all three wire planes | `width_plots()` |
| `numpysaver` | Renders 2-D frame images from NumPy `.npz` files produced by `NumpySaver` WCT nodes | `plot()` |

## Dependencies

| Import | Used by |
|--------|---------|
| `wirecell.util.units` | `morse` — converts tick periods to microseconds |
| `wirecell.gen.morse` | `morse` — `scale_slice`, `patch_chan_mask`, `gauss` |
