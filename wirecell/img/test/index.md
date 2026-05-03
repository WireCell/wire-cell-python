---
generated: 2026-05-02
source-hash: 0000000000000000
---

# wirecell/img/test

Test suite for `wirecell.img` modules.

## Modules

| Module | Purpose | Key Symbols |
|--------|---------|-------------|
| `test_tap.py` | Regression tests for `wirecell.img.tap.PgFiller.add_blob`; verify per-blob corners are sliced to `ncorners` so zero-padded slots do not leak into `node['corners']` | `test_corners_truncated_to_ncorners`, `test_corners_start_column_prepended`, `test_corners_unaffected_when_ncorners_is_max`, `test_corners_per_blob_independent_truncation` |

## Dependencies

- `wirecell.img.tap` — module under test (`pg2nx`, `PgFiller`, `node_types`, `edge_types`)
