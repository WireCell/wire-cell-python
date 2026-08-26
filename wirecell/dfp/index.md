---
generated: 2026-04-29
source-hash: 47d3c81dff866280
---

# wirecell/dfp/

Support for Wire Cell Data Flow Programming (DFP) graphs. Provides tools for constructing, validating, and visualizing directed multigraphs that represent WireCell node connectivity, along with serialization to WireCell JSON configuration format.

## Modules

| Module | Purpose | Key Symbols |
|--------|---------|-------------|
| `__init__.py` | Package marker | — |
| `graph.py` | Build and validate DFP graphs; serialize to WireCell config JSON | `Graph`, `key`, `dekey`, `connect`, `validate`, `wirecell_graph` |
| `nodetype.py` | Represent WireCell `INode` class metadata | `NodeType`, `make`, `loads` |
| `dot.py` | Render DFP graphs as GraphViz/DOT visualizations | `gvgraph`, `gvgraph_nodetypes`, `nodetype_label` |

## Dependencies

| Dependency | Role |
|------------|------|
| `networkx` | Underlying directed multigraph (`nx.MultiDiGraph`) used in `graph.py` |
| `pygraphviz` | GraphViz rendering backend used in `dot.py` and `graph.py` |
