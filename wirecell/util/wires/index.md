---
generated: 2026-04-29
source-hash: 110fe34334d57878
---

# wirecell/util/wires/

Wire geometry description and connectivity modeling for Wire Cell detectors (protoDUNE, DUNE FD, MicroBooNE, ICARUS, DUNE-VD). Provides a schema for storing wire endpoint data, loaders for various text geometry formats, a NetworkX-based connectivity graph for APA electronics, and analysis/plotting utilities.

## Modules

| Module | Purpose | Key Symbols |
|--------|---------|-------------|
| `schema.py` | Namedtuple schema for wire geometry (Point, Wire, Plane, Face, Anode, Detector, Store) and a factory | `Store`, `maker()`, `wire_plane_id()`, `plane_face_apa()` |
| `apa.py` | DUNE APA connectivity: chip-channel-layer mapping, graph construction, Plex class | `Description`, `Plex`, `graph()`, `channel_hash()`, `channel_unhash()` |
| `generator.py` | Geometry primitives and wire wrapping algorithms for rectangular APAs | `Rectangle`, `wrapped_from_top()`, `wrapped_from_top_oneside()`, `onesided_wrapped()` |
| `graph.py` | NetworkX graph traversal and conversion utilities for APA connectivity graphs | `nodes_by_type()`, `neighbors_by_type()`, `parent()`, `to_schema()`, `flatten_to_conductor()` |
| `persist.py` | Serialize/deserialize schema Store objects to/from JSON; detector registry lookup | `load()`, `dump()`, `todict()`, `fromdict()` |
| `array.py` | NumPy array operations on wire endpoint data: pitch, rotation, translation | `endpoints_from_schema()`, `mean_wire_pitch()`, `rotation()`, `translation()` |
| `info.py` | Summary statistics, bounding boxes, pitch analysis, and Jsonnet volume generation | `summary()`, `summary_dict()`, `pitch_summary()`, `jsonnet_volumes()` |
| `plot.py` | Matplotlib wire visualization: per-plane, per-channel, and full-detector plots | `oneplane()`, `allplanes()`, `select_channels()` |
| `common.py` | Shared Wire namedtuple and bounding box helper | `Wire`, `bounding_box()` |
| `db.py` | SQLAlchemy ORM schema for detector connectivity (experimental) | `Detector`, `Anode`, `Board`, `Chip`, `Channel`, `session()` |
| `multitpc.py` | Loader for multi-TPC larsoft text wire geometry files (protoDUNE, SBND, DUNE-VD) | `load()` |
| `onesided.py` | Loader for single-sided celltree wire geometry files (MicroBooNE) | `load()` |
| `icarustpc.py` | Loader for ICARUS multi-TPC wire geometry files | `load()` |
| `dunevd.py` | Loader for DUNE vertical-drift wire geometry files; TPC merging utilities | `load()`, `merge_tpc()`, `merge_wires()` |
| `dune.py` | DUNE APA connectivity graph (partial/alternate implementation) | `ApaConnectivity`, `flatten_cclsm()` |
| `regions.py` | Wire region descriptions and shorted-wire handling (MicroBooNE) | `Border`, `Region`, `uboone_shorted()` |
| `summary.py` | Legacy wire summary printing utility | `wire_summary()` |

## Dependencies

| Import | Purpose |
|--------|---------|
| `wirecell.units` | Physical unit conversions applied to all wire coordinates |
| `wirecell.util.detectors` | Canonical detector name resolution for wires files |
| `wirecell.util.jsio` | JSON I/O with compression support used by `persist` |
| `wirecell.util.geo.shapes` | `Ray`, `Point3D` used in `common` |
