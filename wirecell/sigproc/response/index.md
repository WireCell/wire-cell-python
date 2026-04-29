---
generated: 2026-04-29
source-hash: c48d48257d65f393
---

# wirecell/sigproc/response/

Tools for representing, converting, and manipulating wire-cell field and electronics response functions. Supports multiple data representations (schema, POD, numpy arrays) and serialization formats (JSON, compressed JSON, NPZ).

## Modules

| Module | Purpose | Key Symbols |
|--------|---------|-------------|
| `__init__` | Core response function math and data structures | `ResponseFunction`, `electronics`, `convolve`, `normalize`, `average`, `field_response_spectra`, `plane_impact_blocks`, `rf1dtoschema`, `line` |
| `schema` | Dataclass schema mirroring WCT C++ field response types | `FieldResponse`, `PlaneResponse`, `PathResponse`, `asdict` |
| `arrays` | Numpy array representation of field responses | `toarray`, `toschema`, `coldelec`, `fr2arrays`, `pr2array` |
| `persist` | Serialization/deserialization across all representations and file formats | `load`, `dump`, `loads`, `dumps`, `topod`, `toarray`, `schema2pod`, `pod2schema` |
| `plots` | Matplotlib visualization of field responses | `plot_planes`, `plot_conductors`, `plot_specs`, `lg10`, `get_current` |

## Dependencies

- `wirecell.units` — physical unit constants used throughout
- `wirecell.util` — `unitify`, `detectors`, `jsio`
- `wirecell.sigproc.response.schema` — imported by `__init__`, `arrays`, `persist`
