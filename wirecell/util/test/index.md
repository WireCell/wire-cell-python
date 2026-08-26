---
generated: 2026-04-29
source-hash: 7bd400c700d2fdcd
---

# wirecell/util/test

Test suite for the `wirecell.util` package, covering wire geometry, database connectivity, signal processing utilities, path resolution, and tree data structures. Tests range from unit checks to integration tests that generate diagnostic plots and files.

## Modules

| Module | Purpose | Key Symbols |
|--------|---------|-------------|
| `_test_db.py` | Tests ORM database connectivity for detector/crate/wib/board/chip/channel hierarchy | `test_classes`, `db.session`, `db.Detector`, `db.DetectorCrateLink` |
| `_test_full.py` | Integration test building full DUNE detector connectivity without geometry | `test_full`, `chip_conductor_matrix`, `flatten_chip_conductor_map` |
| `plot_impactresponse.py` | ROOT-based plotting script for impact response histograms with bilog scaling | `bilogify`, `set_palette`, `draw_wires` |
| `test-paths.py` | Tests `wirecell.util.paths` listify and resolve utilities | `test_listify`, `test_resolve`, `test_resolve_no_wirecell_path`, `test_resolve_with_wirecell_path` |
| `test_cont_conv.py` | Placeholder for continuous convolution method tests | _(empty)_ |
| `test_convo.py` | Exercises frequency- and time-domain convolution via `lmn` signals and plots results | `tconvolve`, `convolve`, `gauss`, `sig` |
| `test_detectors.py` | Tests detector registry path resolution for known detectors | `test_environment`, `test_resolve`, `test_resolve_list`, `test_resolve_missing` |
| `test_tdm.py` | Tests `tdm.Tree` node creation, deep path access, metadata, and array storage | `test_tree_empty`, `test_tree_deep`, `test_tree_metadata`, `test_tree_array` |
| `test_wireplots.py` | Generates PDF wire geometry plots per plane, chip, board, and WIB using NetworkX | `test_plot_plane_chip`, `test_plot_conductor`, `test_plot_chip`, `test_plot_board`, `test_plot_wib_wires`, `test_plot_wib` |
| `test_wires_array.py` | Tests wire endpoint loading, correction, rotation/translation transforms, and plots | `test_load_correct_transform` |

## Dependencies

- `wirecell.util.wires.db` — ORM models for detector hierarchy
- `wirecell.util.wires.{apa,graph,schema,info,array,persist}` — wire geometry graph and array utilities
- `wirecell.util.paths` — `listify`, `resolve`
- `wirecell.util.tdm` — `Tree` data structure
- `wirecell.util.detectors` — detector registry and file resolution
- `wirecell.util.lmn` — sampled signal and resampling utilities
- `wirecell.util.plottools` — `pages` context manager for PDF output
- `wirecell.resp.plots` — `plot_signals`, `multiply_period`
