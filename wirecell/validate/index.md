---
generated: 2026-04-29
source-hash: a5872d411d27eb80
---

# wirecell/validate

Provides validation tools for Wire Cell processing, comparing and visualizing Magnify ROOT histograms against reference outputs. Includes array rebinning utilities, custom matplotlib colormaps, and a ROOT file helper layer for loading, diffing, and summarizing histograms.

## Modules

| Module | Purpose | Key Symbols |
|--------|---------|-------------|
| `arrays` | ndarray rebinning utilities | `rebin`, `bin_ndarray` |
| `cmaps` | Custom matplotlib colormap definitions and registration | `blue_red1`, `blue_red2`, `cdict1–4` |
| `plots` | Validation plot generators for Magnify-style data | `three_horiz`, `one_plane`, `channel_summaries`, `make_cmaps` |
| `root` | ROOT file I/O and histogram introspection helpers | `open_file`, `load_obj`, `is_hist`, `hist_to_dict`, `resize_hist2f` |

## CLI Commands

| Command | Description |
|---------|-------------|
| `diff-hists` | Produce an output ROOT file holding the difference of named histograms from two input files |
| `magnify-diff` | Form a new Magnify file holding histograms that are the difference of those from two inputs |
| `magnify-jsonify` | Serialize summary info about all histograms in a Magnify file to JSON |
| `magnify-dump` | Dump Magnify histograms into NumPy `.npz` files with bin-edge arrays |
| `npz-load` | Load and subtract two `.npz` files, writing the result to a compressed output |
| `magnify-plot-reduce` | Reduce a Magnify 2D histogram along the time axis and plot per-channel summaries |
| `magnify-plot` | Plot Magnify histograms for U/V/W planes with optional rebinning, baseline subtraction, thresholding, and saturation |

## Dependencies

- `wirecell.units` — physical unit constants used in CLI context
- `wirecell.util.cli` — shared Click context and logging setup (`context`, `log`)
- `wirecell.validate.plots` — `channel_summaries`, `three_horiz` (used by plot subcommands)
- `wirecell.validate.arrays` — `rebin`, `bin_ndarray` (used by `magnify-plot`)
