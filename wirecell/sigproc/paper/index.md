---
generated: 2026-04-29
source-hash: 9d29044a6b5a51bd
---

# wirecell/sigproc/paper

Helper scripts for generating figures and data files used in signal processing papers. Currently focused on digitized ADC response plots derived from Garfield simulation data for MicroBooNE-style wire geometries.

## Modules

| Module | Purpose | Key Symbols |
|--------|---------|-------------|
| `noise` | Generate ADC response figures and text data files from Garfield wire response data | `figure_adc`, `filter_response_functions` |

## CLI Commands

| Command | Description |
|---------|-------------|
| `python noise.py [garfield_tarball] [outname]` | Generate digitized ADC response plots (PDF) and data tables (TXT) from a Garfield tarball |

## Dependencies

| Dependency | Role |
|------------|------|
| `wirecell.units` | Physical unit constants (e.g. `eplus`, `us`) |
| `wirecell.sigproc.garfield` | Load Garfield simulation data from tarball |
| `wirecell.sigproc.response` | Compute line responses from raw Garfield data |
| `wirecell.sigproc.plots` | Render digitized line plots |
