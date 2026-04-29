---
generated: 2026-04-29
source-hash: de98b5a88084aa00
---

# wirecell/plot

Provides plotting utilities for WireCell Toolkit frame data, including waveform visualization, spectral analysis, channel correlation, and frame comparison. Exposes a `wirecell-plot` CLI for generating PDF or image outputs from ario frame files. Supports multiple data tiers (raw, gauss, wiener) and configurable rebaselining transforms.

## Modules

| Module | Purpose | Key Symbols |
|--------|---------|-------------|
| `__main__` | CLI entry point for `wirecell-plot` | `cli`, `ntier_frames`, `frame`, `comp1d`, `channels` |
| `frames` | Core frame plotting functions | `wave`, `spectra`, `comp1d`, `channel_correlation`, `frame_means`, `common_channels`, `select_channels` |
| `rebaseline` | Frame rebaselining transforms | `median`, `mean`, `ac`, `none` |
| `cmaps` | Colormap selection utilities | `good`, `tier` |

## CLI Commands

| Command | Description |
|---------|-------------|
| `ntier-frames` | Plot a number of per tier frames from multiple ario frame files |
| `frame` | Visualize a WCT frame with a named plotter (wave or spectra) |
| `comp1d` | Compare waveforms or spectra across multiple frame files |
| `channel-correlation` | Plot channel-to-channel correlation matrix |
| `frame-diff` | Compute and image the difference between two frames |
| `frame-image` | Dump a frame array directly to an image file |
| `frame-means` | Plot frame with per-channel and per-tick projected means |
| `digitizer` | Plot ADC digitizer rounding comparison from test_digitizer JSON |
| `channels` | Plot waveforms and spectra for selected channels across files |
| `channels-shift` | Plot channel waveforms with shift analysis using LMN signals |

## Dependencies

| Import | Role |
|--------|------|
| `wirecell.util.ario` | Loading ario-format frame data files |
| `wirecell.util.plottools` | PDF/image output helpers (`PdfPages`, `NameSequence`, `pages`, `image`, `imopts`) |
| `wirecell.util.cli` | CLI decorators (`context`, `frame_input`, `image_output`, `jsonnet_loader`) |
| `wirecell.util.frames` | Frame loading (`load`) |
| `wirecell.util.lmn` | Signal/sampling objects used in `channels-shift` |
| `wirecell.resp.plots` | `plot_shift` for shift analysis |
| `wirecell.units` | Physical unit constants and scaling |
