---
generated: 2026-04-29
source-hash: 3ae91f64e5909482
---

# wirecell/img

Tools for Wire-Cell Toolkit imaging: loading and inspecting cluster graphs, converting them to visualization formats (ParaView VTK, Bee JSON), and producing diagnostic plots of blobs, activity, and depositions. The package bridges WCT's internal cluster array format and standard graph/plotting libraries.

## Modules

| Module | Purpose | Key Symbols |
|--------|---------|-------------|
| `tap` | Load cluster files (JSON or archive) into NetworkX graphs | `load()`, `load_ario()`, `load_jsio()`, `pg2nx()`, `make_pggraph()` |
| `clusters` | Index and query cluster graph nodes by type code | `ClusterMap` |
| `converter` | Convert cluster/depo data to VTK/ParaView objects and sample blob volumes | `undrift_blobs()`, `undrift_depos()`, `clusters2blobs()`, `blobpoints()`, `blob_uniform_sample()` |
| `plots` | 2D histogram and matplotlib plotting helpers for activity and blob masks | `activity()`, `blobs()`, `mask_blobs()`, `wire_blob_slice()`, `Hist2D` |
| `plot_blobs` | Per-graph blob diagnostic plots (coordinate histograms, view projections) | `plot_x/y/z/t()`, `plot_views()`, `plot_tx/ty/tz()` |
| `plot_depos_blobs` | Combined depo+blob overlay plots | `plot_xz()`, `plot_outlines()`, `plot_views()` |
| `dump_blobs` | Text dump of per-blob signatures for debugging | `bsignature()`, `dump_blobs()` |
| `dump_bb_clusters` | Text dump of blob-cluster signatures | `csignature()`, `dump_bb_clusters()` |
| `anidfg` | Animate a TbbFlow data-flow graph from log output | `parse_log()`, `generate_graph()`, `render_graph()` |

## CLI Commands

| Command | Description |
|---------|-------------|
| `plot-depos-blobs` | Plots combining depos and blobs |
| `plot-blobs` | Produce plots related to blobs in cluster |
| `dump-blobs` | Dump blob signatures in cluster to a file |
| `dump-bb-clusters` | Dump blob cluster signatures |
| `inspect` | Inspect a cluster file |
| `paraview-blobs` | Convert a cluster file to ParaView `.vtu` files of blobs |
| `paraview-activity` | Convert cluster files to ParaView `.vti` files of activity |
| `paraview-depos` | Convert WCT depo file to a ParaView `.vtp` file |
| `bee-blobs` | Produce a Bee JSON file from a cluster file |
| `bee-flashes` | Produce a Bee JSON file from a flash ITensorSet file |
| `activity` | Plot activity from a cluster file |
| `blob-activity-stats` | Return statistics on blob and activity (totals, fractions found/missed) |
| `blob-activity-mask` | Plot blobs as masks on channel activity |
| `wire-slice-activity` | Plot the activity in one slice as wires and blobs |
| `anidfg` | Produce an animated graph visualization from a TbbFlow DFG log |
| `transform-depos` | Apply spatial transformations (rotate, translate) to depo distributions |

## Dependencies

| Import | Role |
|--------|------|
| `wirecell.units` | Physical unit constants used throughout coordinate conversions |
| `wirecell.util.ario` | Archive I/O for reading cluster and depo files |
| `wirecell.util.jsio` | JSON/JSONNET loading for cluster files |
| `wirecell.util.plottools.pages` | Multi-page PDF output helper |
| `wirecell.util.wires.schema.plane_face_apa` | Wire-plane ID decomposition used in plots |
| `wirecell.gen.depos` | Deposition file streaming (`deposmod.stream`) |
