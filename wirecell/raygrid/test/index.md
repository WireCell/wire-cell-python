---
generated: 2026-04-29
source-hash: cee2fd230b0ddaac
---

# wirecell/raygrid/test

Test suite for the `wirecell.raygrid` package, covering geometric primitives, coordinate systems, tiling algorithms, and visualization utilities. Tests use `pytest` and `torch` to validate correctness of ray-grid computations including crossing points, pitch locations, blob building, and combinatorial helpers.

## Modules

| Module | Purpose | Key Symbols |
|---|---|---|
| `sampledata` | Shared tensor fixtures for blob and crossing data used across tests | `all_blobs`, `all_crossings` |
| `test_coordinates` | Tests `Coordinates` class: zero crossings, ray crossings, pitch locations, C++ dump | `test_coordinates`, `test_three`, `test_dump_coordinates` |
| `test_examples` | Tests `symmetric_views` geometry and renders a PDF of view vectors | `test_symmetric_views` |
| `test_funcs` | Unit tests for geometric primitives: vector, direction, crossing, pitch | `test_vector`, `test_direction`, `test_crossing`, `test_pitch` |
| `test_plots` | Integration tests for activity filling and blob-building plot pipeline | `test_activity_fill`, `test_blob_building`, `make_stuff` |
| `test_tiling` | Tests individual tiling steps, trivial tiling, crossings, and raster track projection | `test_individual_steps`, `test_simple_tiling`, `test_fresh_crossings` |
| `test_uncombo` | Tests `get_unchosen_elements` for combinatorial complement computation | `test_uncombo` |

## Dependencies

- `wirecell.raygrid.coordinates` — `Coordinates`
- `wirecell.raygrid.examples` — `symmetric_views`, `random_points`, `random_groups`, `fill_activity`, `Raster`
- `wirecell.raygrid.funcs` — `vector`, `ray_direction`, `crossing`, `pitch`, `get_unchosen_elements`
- `wirecell.raygrid.tiling` — `trivial`, `blob_crossings`, `blob_insides`, `blob_bounds`, `apply_activity`
- `wirecell.raygrid.plots` — `make_fig_2d`, `point_activity`, `blobs_strips`, `blobs_corners`, `save_fig`, `PdfPages`
