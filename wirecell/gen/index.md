---
generated: 2026-04-29
source-hash: 14007a1d552b563c
children-hash: a0af4b009132c127
---

# wirecell/gen

Signal simulation utilities for Wire Cell Toolkit. Provides tools for generating, transforming, and visualizing ionization depositions ("depos") in various geometric patterns (point, line, sphere, morse grid), along with frame inspection, noise spectrum modeling, and line-track generation aligned to detector wire planes.

## Modules

| Module | Purpose | Key Symbols |
|--------|---------|-------------|
| `depogen` | Generate structured sets of depos: random line tracks within a bounding box, and spherical shell patterns | `lines()`, `sphere()`, `track_info_types` |
| `depos` | Load, save, transform, and plot deposition arrays in WCT NPZ/JSON formats | `load()`, `dump()`, `stream()`, `apply_units()`, `move()`, `center()`, `plot_qxz()` |
| `linegen` | Generate single line-track depo sets aligned to wire-plane or global angles, with full track metadata | `TrackConfig`, `TrackMetadata`, `generate_line_track()`, `generate_and_save_line_track_in_detector()` |
| `morse` | Generate and analyze the "morse" depo pattern (2D grids targeting each wire plane) and fit Gaussian peak widths | `generate()`, `frame_peaks()`, `wave_peaks()`, `WavePeak`, `FramePeaks` |
| `noise` | Noise spectrum modeling: Rayleigh/Gaussian spectra, interpolation, extrapolation, aliasing, and resampling | `Spec`, `Collect`, `fictional()`, `gaussian_spec()` |
| `plots` | Legacy frame visualization from NumpySaver NPZ files | `numpy_saver()` |
| `sim` | Load and plot WCT simulation frames and depo sets from NPZ files | `Frame`, `Depos`, `baseline_subtract()`, `group_channel_indices()` |
| `plots/` | Visualization utilities for morse patterns and NumpySaver frames | `width_plots()` |
| `test/` | Tests and diagnostic plots for linegen, noise, and depo utilities | — |

## CLI Commands

| Command | Description |
|---------|-------------|
| `unitify-depos` | Set units for a WCT JSON deposition file |
| `depos-bb` | Report bounding box of entries in a depos file |
| `shift-depos` | Move depos to given center and/or apply relative shift |
| `move-depos` | Apply transformations to a JSON depos file and create a new file |
| `plot-depos` | Make a plot from a WCT depo file |
| `plot-test-boundaries` | Make plots from the boundaries test NPZ output |
| `plot-sim` | Make plots of sim quantities saved into numpy array files |
| `depo-morse` | Produce depos in a morse pattern aligned to wires of each plane |
| `morse-summary` | Find signal processing smearing from morse signal NPZ |
| `morse-splat` | Output DepoFluxSplat configuration from a morse summary |
| `morse-plots` | Produce plots for analysis of the morse track pattern |
| `depo-line` | Generate a single line of depos between two endpoints |
| `depo-lines` | Generate random ideal line-source tracks of depos |
| `depo-point` | Generate a single point depo |
| `depo-sphere` | Generate ideal spherical shell of depos |
| `frame-stats` | Print stats on the time distribution of a frame |
| `linegen` | Generate a line of depos given wire-plane rotation angles |
| `detlinegen` | Generate a line of depos for a specific named detector and APA |

## Dependencies

| Import | Used by |
|--------|---------|
| `wirecell.units` | `depogen`, `depos`, `linegen`, `morse`, `noise`, `sim` — physical unit constants |
| `wirecell.util.ario` | `depos` — generic array file I/O |
| `wirecell.util.functions` | `__main__`, `linegen` — `unitify`, `unitify_parse` |
| `wirecell.util.wires.array` | `linegen`, `morse` — wire endpoint arrays, `mean_wire_pitch`, rotation |
| `wirecell.util.wires.persist` | `linegen`, `__main__` — load wire store from canonical detector name |
| `wirecell.util.cli` | `__main__` — `context`, `log`, `frame_input`, `image_output` |
| `wirecell.util.plottools` | `__main__` — `pages` multi-page PDF output |
| `wirecell.lmn` | `noise` — `hermitian_mirror` for spectral symmetry |
| `wirecell.gen.morse` | `__main__`, `plots/morse` — morse pattern generation and peak analysis |
| `wirecell.gen.sim` | `__main__` — `Frame`, `Depos`, channel grouping |
| `wirecell.gen.depogen` | `__main__` — `lines`, `sphere` |
| `wirecell.gen.linegen` | `__main__` — `TrackConfig`, track generation functions |
