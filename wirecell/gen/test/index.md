---
generated: 2026-04-29
source-hash: a52564919d0c6972
---

# wirecell/gen/test

Test and diagnostic scripts for the `wirecell.gen` package, covering deposition data visualization, impact response inspection, line generator geometry, and noise spectrum utilities. Includes both pytest-based unit tests and standalone plotting scripts.

## Modules

| Module | Purpose | Key Symbols |
|---|---|---|
| `plot_g4tuple.py` | Load and plot Geant4 deposition tuples (dE/dX, dN/dX, 2D maps) | `load_depos`, `plot_dedx`, `plot_dndx`, `plot_nxz` |
| `plot_impactzipper.py` | ROOT-based visualization of impact zipper UVW histograms with bilog color scaling | `bilogify`, `set_palette` |
| `test_linegen.py` | pytest tests for TPC angle/direction conversion and rotation matrix orthogonality | `test_round_trip`, `test_wplane_is_global`, `test_select_angles`, `test_orthogonal_rotation` |
| `test_noise.py` | Visual regression tests for noise spectrum interpolation, extrapolation, aliasing, and resampling | `Noise`, `doit`, `spec_wids` |

## Dependencies

| Import | Purpose |
|---|---|
| `wirecell.units` | Unit constants (`degree`, `radian`) used in angle tests |
| `wirecell.gen.linegen` | Direction/angle conversion functions under test |
| `wirecell.gen.noise` | Noise spectrum classes and functions under test (`Collect`, `Spec`, `gaussian_waves`, etc.) |
