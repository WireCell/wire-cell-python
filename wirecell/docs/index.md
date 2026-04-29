---
generated: 2026-04-29
source-hash: b35aad5f34a43619
---

# wirecell/docs

Provides tooling to generate, check, and display `index.md` documentation files for every Python module directory in wire-cell-python. It computes content hashes to detect staleness and can emit LLM prompts (single-dir, combined, or shell script) to regenerate out-of-date indexes.

## Modules

| Module | Purpose | Key Symbols |
|--------|---------|-------------|
| `__init__.py` | Package marker and module docstring | — |
| `__main__.py` | CLI entry point and all supporting logic | `cli`, `check_cmd`, `prompt_cmd`, `apply_cmd`, `show_cmd`, `_find_module_dirs`, `_source_hash`, `_children_hash`, `_single_prompt`, `_combined_prompt`, `_shell_script` |

## CLI Commands

| Command | Description |
|---------|-------------|
| `check` | Show which index.md files are missing or stale; exits 1 if any need regeneration |
| `prompt` | Emit an LLM prompt or shell script to regenerate stale index.md files (single-dir, combined, or `$WCPY_LLM` script) |
| `apply` | Parse combined LLM output (option B FILE blocks) and write each index.md to disk |
| `show` | Display an index.md with terminal colour formatting or as JSON |

## Dependencies

| Import | Role |
|--------|------|
| `wirecell.util.cli` | `context`, `log` — shared CLI group factory and logging |
