---
generated: 2026-04-29
source-hash: 5829385583e0bcfe
---

# wirecell/util/

General-purpose utilities for Wire-Cell Toolkit Python code: file I/O (JSON/Jsonnet, numpy archives, detector registries), unit conversion, signal resampling, frame and wire geometry manipulation, peak finding, and plotting helpers. It also provides the `wcpy util` CLI entry point with commands spanning wire geometry conversion, frame processing, and data-format inspection.

## Modules

| Module | Purpose | Key Symbols |
|--------|---------|-------------|
| `ario` | Read-only dict-like access to archive files (.npz, .zip, .tar) with lazy loading | `Tar`, `Zip`, `load`, `stem_if` |
| `bbox` | Bounding-box index utilities for numpy array slicing | `union`, `union_slice`, `union_array` |
| `cdm` | Cluster data model file inspection | `looks_like`, `dumps` |
| `channels` | Parse channel-range specification strings | `parse_range` |
| `cli` | Click decorators for building WCT sub-commands | `context`, `jsonnet_loader`, `frame_input`, `image_output`, `anyconfig_file` |
| `codec` | JSON encoder + dataclass serialization helpers | `JsonEncoder`, `json_dumps`, `dataclass_dictify` |
| `detectors` | Detector registry lookup via `detectors.jsonnet` | `resolve`, `load` |
| `fileio` | Low-level file I/O: WIRECELL_PATH, zip/tar/dir iterators | `wirecell_path`, `load`, `zipball`, `tarball`, `dirball` |
| `frame_split` | Split multi-APA frame arrays into per-plane .npz files | `apa`, `protodune`, `save_one`, `offset_cols`, `rebin_cols` |
| `frames` | Frame dataclass and ario-based frame loader | `Frame` |
| `functions` | WCT system-of-units evaluation | `unitify`, `unitify_parse` |
| `jsio` | Uniform JSON/Jsonnet loading with path and TLA support | `load`, `resolve`, `tla_pack`, `wash_path` |
| `lmn` | LMN rational resampling method for signals | `Sampling`, `Signal`, `interpolate`, `hermitian_mirror`, `rational_size` |
| `paths` | WCT file-path resolution and WIRECELL_PATH helpers | `resolve`, `listify`, `flatten` |
| `peaks` | Peak finding and representation in 1D/2D waveform arrays | `Peak`, dataclass peak types |
| `plottools` | Matplotlib multi-page output and axis helpers | `pages`, `NameSequence`, `NameSingleton`, `rescaley` |
| `points` | PCA and point-cloud geometry utilities | `pca_eigen` |
| `tdm` | WCT Tensor Data Model: load/dump/convert TDM archive files | `Tree`, `load`, `dumps`, `tohdf`, `pc2vtk` |

## CLI Commands

| Command | Description |
|---------|-------------|
| `convdown` | Calculate sizes for simultaneous convolution and downsample |
| `lmn` | Print LMN resampling parameters for a given source and target sampling |
| `convert-oneside-wires` | Convert onesided wire geometry text file to WCT JSON format |
| `convert-multitpc-wires` | Convert multi-TPC wire geometry text file to WCT JSON format |
| `convert-icarustpc-wires` | Convert ICARUS TPC wire geometry text file to WCT JSON format |
| `convert-dunevd-wires` | Convert DUNE-VD wire geometry text file to WCT JSON format |
| `convert-uboone-wire-regions` | Convert MicroBooNE shorted-wire CSV to WCT wire-region JSON |
| `plot-wire-regions` | Plot wire regions as polygons to a PDF |
| `wires-info` | Print summary of a wires JSON file |
| `wires-ordering` | Plot wire ordering per plane to a PDF |
| `wires-channels` | Plot wire positions vs channel IDs to a PDF |
| `wires-volumes` | Print a geometry JSON fragment for a wires file |
| `plot-wires` | Plot wires from a WCT JSON wire file |
| `plot-select-channels` | Plot wires for selected channel IDs |
| `gen-plot-wires` | Generate synthetic wires and plot them |
| `make-wires` | Generate a WCT wires file for a named detector (apa) |
| `make-map` | Generate a WCT channel map file as .npz or LaTeX |
| `gravio` | Write a Graphviz dot file of APA connectivity |
| `make-wires-onesided` | Generate a WCT wires file with configurable onesided geometry |
| `wire-channel-map` | Debug: print (plane, channel) → wire index mapping |
| `wire-summary` | Write a JSON summary of wires geometry |
| `channel-summary` | Print per-detector/anode/face/plane channel counts |
| `frame-split` | Split a frame archive into per-plane .npz files |
| `npz-to-img` | Render a 2D numpy array in an .npz file as an image |
| `ls` | List contents of a WCT archive file (.npz, .zip, .tar) |
| `pc2pd` | Convert a WCT TDM point-cloud file to VTK PolyData (.vtp) |
| `tdm2hdf` | Convert a WCT TDM archive file to HDF5 |
| `dump-tdm` | Print the tensor-data-model structure of a file |
| `npz-to-wct` | Convert a plain .npz frame array to WCT frame format |
| `ario-cmp` | Compare two ario archives for identical contents |
| `detectors` | List canonical detectors known via the WIRECELL_PATH registry |
| `resample` | Resample all frames in a frame file to a new tick period |
| `resolve` | Resolve a file or detector name to a path via WIRECELL_PATH |
| `frame-block` | Build a dense (padded/cropped) frame block from a frame file |
| `framels` | Print shape, channel range, and tick info for a frame file |

## Dependencies

| Dependency | Used for |
|------------|---------|
| `wirecell.units` | System-of-units constants evaluated by `unitify` |
