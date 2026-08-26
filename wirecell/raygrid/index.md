---
generated: 2026-04-29
source-hash: 1c0b4f89ddfa4560
children-hash: 755a0aa224b1d588
---

# wirecell/raygrid

Implements a ray-grid tomographic tiling algorithm for 2D scalar field reconstruction from multi-view wire-plane activity data. Provides coordinate transforms between ray-grid and Cartesian space, blob-finding via iterative view intersection, and visualization utilities for wire geometry and tiling results.

## Modules

| Module | Purpose | Key Symbols |
|---|---|---|
| `activity` | Threshold a 1D activity tensor into half-open contiguous ranges | `threshold_1d` |
| `coordinates` | Ray-grid coordinate system: pitch, crossing, and index transforms across views | `Coordinates`, `coordinates_from_wires` |
| `examples` | Test fixtures: symmetric view geometry, random points, clumpy groups, activity fill, 3D raster | `symmetric_views`, `random_points`, `random_groups`, `fill_activity`, `Raster` |
| `funcs` | Geometric primitives: ray pitch, crossing point, direction vectors, combinatorial complement | `pitch`, `crossing`, `ray_direction`, `vec_direction`, `combo_partition`, `get_unchosen_elements` |
| `plots` | Matplotlib visualization of activity strips, blob corners, and 2D wire projections | `make_fig_2d`, `point_activity`, `blobs_strips`, `blobs_corners`, `save_fig` |
| `tiling` | Core blob-finding: apply per-view activities to iteratively refine blobs | `Tiling`, `trivial`, `trivial_blobs`, `apply_activity`, `blob_crossings`, `blob_insides`, `blob_bounds`, `strip_pairs`, `get_true_runs`, `expand_blobs_with_activity` |
| `wires` | Load wire endpoint arrays from detector schema and construct ray-grid views | `load`, `make_views`, `to2d`, `to_coordinates` |

## CLI Commands

| Command | Description |
|---|---|
| `plot-cells` | Plot wire plane cell projections in 2D for a given detector and anode index |

## Dependencies

- `wirecell.util.cli` — `log`, `context`
- `wirecell.util.wires.array` — `correct_endpoint_array`, `mean_wire_pitch`, `endpoints_from_schema`
- `wirecell.util.wires.persist` — `load`
