# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment and commands

This project is managed with [uv](https://github.com/astral-sh/uv).

```sh
uv sync                  # install core dependencies into .venv
uv sync --all-extras     # also install optional torch/dnn dependencies
source .venv/bin/activate  # or prefix every command with `uv run`
```

The unified CLI entry point is `wcpy` (also aliased as `wirecell`):

```sh
wcpy --help
wcpy help all            # show every namespace and its commands
wcpy help wirecell.util  # show just the util namespace (dot or slash notation)
```

Run tests with the system `pytest` (the venv pytest is currently broken due to a missing `iniconfig` dependency):

```sh
pytest test/                                  # top-level test suite
pytest wirecell/util/test/test_tdm.py         # single test file
pytest wirecell/util/test/test_tdm.py::test_tree_empty  # single test
```

Build a wheel:

```sh
uv build
```

## Architecture

### Package layout

The `wirecell/` directory is both the Python package root and the git repo root (non-standard `src`-less layout declared in `pyproject.toml`).

Each functional area lives in a sub-package:

| Sub-package | Domain |
|-------------|--------|
| `util` | General I/O, LMN resampling, wire geometry, TDM, frame helpers, plotting |
| `resp` | Field/electronics response ingestion, resampling, Garfield file parsing |
| `sigproc` | Signal processing, noise models, ForWaRD simulation |
| `gen` | Deposition and frame simulation (tracks, noise spectra) |
| `img` | Cluster graph loading and conversion to VTK/Bee formats |
| `raygrid` | Ray-grid tiling and wire-plane coordinate tools |
| `dfp` | Data-flow programming graph construction (NetworkX + GraphViz) |
| `pgraph` | WCT config → GraphViz dot visualization |
| `bee` | Bee visualization server file helpers |
| `docs` | `index.md` generation, staleness checking, LLM prompt emission |
| `dnn` / `pytorch` | PyTorch DNN training (optional; requires `torch` extra) |

### CLI convention

Every sub-package exposes a Click command group via `__main__.py`:

```python
from wirecell.util.cli import context, log

@context("myns")          # registers the group as "wcpy myns"
def cli(ctx):
    """Docstring becomes the namespace help text."""
    pass

@cli.command("my-cmd")
def cmd_my_cmd():
    """Per-command help text."""
    log.info("use log, not print()")
```

`wirecell/__main__.py` imports each sub-package's `cli` object and attaches it to the root group. Adding a new namespace means adding it to the `subs` string there.

The `@context` decorator automatically adds `-l/--log-output` and `-L/--log-level` options to every group. Use the module-level `log` from `wirecell.util.cli` for all output (not `print()`).

CLI path arguments in `wcpy docs` and `wcpy help` accept both slash notation (`wirecell/util`) and dot notation (`wirecell.util`).

### Key shared utilities (`wirecell.util`)

- **`units.py`** — WCT system of units (base unit: millimeter). Always use `from wirecell import units` and express physical quantities as e.g. `3.0 * units.cm`. The `unitify()` function in `util.functions` evaluates unit strings like `"3*cm"`.
- **`jsio.py`** — unified JSON/Jsonnet loader; respects `WIRECELL_PATH` env var and `-J` CLI option for search paths. Use `jsio.load(fname, paths)` rather than raw `json.load`.
- **`ario.py`** — dict-like read-only access to `.npz`, `.zip`, `.tar` archives with lazy loading. `ario.load(path)` is the entry point.
- **`tdm.py`** — WCT Tensor Data Model: `Tree` class models HDF5-style nested tensors; `tdm.load(ario_file)` returns a list of `Tree` objects.
- **`cli.py`** — shared Click decorators: `context`, `jsonnet_loader`, `frame_input`, `image_output`.
- **`lmn.py`** — LMN rational resampling; `lmn.interpolate(signal, new_period)` is the main entry point.
- **`frames.py`** — `Frame` dataclass wrapping (samples, channels, tickinfo) arrays from WCT `.npz` frame files.

### File formats

WCT data flows through several file types:

- **`.npz`** frames: arrays named `frame_<tag>_<index>`, `channels_<tag>_<index>`, `tickinfo_<tag>_<index>`
- **`.json` / `.jsonnet`**: detector configuration and geometry files; resolved via `WIRECELL_PATH`
- **TDM archives** (`.npz`, `.tar`, `.zip`): keys follow `tensorset_<id>` / `tensor_<id>_<idx>_metadata` / `tensor_<id>_<idx>_array` pattern
- **`detectors.jsonnet`** registry (found via `WIRECELL_PATH`): maps canonical detector names to their file paths

### Documentation index system

Each sub-package directory contains an `index.md` with YAML frontmatter tracking `source-hash` and `children-hash`. Use `wcpy docs` to manage these:

```sh
wcpy docs check all                       # check staleness
wcpy docs prompt wirecell/util            # emit LLM prompt for one module
wcpy docs prompt --monolith               # emit combined prompt for all stale modules
wcpy docs prompt --script                 # emit shell script using $WCPY_LLM
wcpy docs apply < llm_output.md          # write FILE blocks from LLM response
wcpy docs show wirecell.util             # display a module's index.md
```
