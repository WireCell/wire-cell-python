#!/usr/bin/env pytest
"""Regression tests for wirecell.img.tap.

Locks the corners-truncation fix (commit fa08955): PgFiller.add_blob
must slice node['corners'] to the per-blob ncorners count and not
expose the zero-padded slots c{ncorners}..c23.
"""

from collections import namedtuple

import numpy as np
import pytest

from wirecell.img import tap


_Node = namedtuple("Node", "x")
_Edge = namedtuple("Edge", "edge_index")

# Column widths from PgFiller.dtypes.
_NCOLS = {"a": 6, "w": 12, "b": 39, "s": 7, "m": 5}


def _empty_pg():
    """Return a pg dict with empty arrays for every node/edge type."""
    pg = {}
    for nc, ntype in tap.node_types.items():
        pg[ntype] = _Node(x=np.zeros((0, _NCOLS[nc])))
    for ec in tap.edge_types:
        nt1 = tap.node_types[ec[0]]
        nt2 = tap.node_types[ec[1]]
        pg[(nt1, ec, nt2)] = _Edge(edge_index=np.zeros((0, 3), dtype=int))
    return pg


def _make_blob_row(desc, ident, start, ncorners, corners_yz):
    """Build one 39-column blob row.

    corners_yz: iterable of (y, z) tuples, length must equal ncorners
    (extra slots up to 12 are zero-filled).
    """
    assert len(corners_yz) == ncorners
    row = np.zeros(39)
    row[0] = desc          # desc
    row[1] = ident         # ident
    row[2] = 1.0           # val
    row[3] = 0.0           # unc
    row[4] = 0             # faceid
    row[5] = 0             # sliceid
    row[6] = start         # start
    row[7] = 2.0           # span
    # cols 8..13 are min1,max1,min2,max2,min3,max3 (left as 0)
    row[14] = ncorners
    # 12 (y,z) pairs in cols 15..38
    for i, (y, z) in enumerate(corners_yz):
        row[15 + 2 * i] = y
        row[15 + 2 * i + 1] = z
    return row


def _build_graph_with_blob_rows(brows):
    pg = _empty_pg()
    pg["blob"] = _Node(x=np.asarray(brows))
    return tap.pg2nx("test", pg)


def test_corners_truncated_to_ncorners():
    """Bug fa08955: corners must be sliced to ncorners, not all 12."""
    real = [(1, 1), (1, 2), (2, 2), (2, 1)]
    row = _make_blob_row(desc=1, ident=10, start=0.0,
                         ncorners=4, corners_yz=real)
    gr = _build_graph_with_blob_rows([row])

    corners = gr.nodes[1]["corners"]
    assert corners.shape == (4, 3), \
        f"expected (4, 3) for ncorners=4, got {corners.shape}"
    # No row is zero in the (y, z) slice.
    yz = corners[:, 1:]
    assert not (yz == 0).all(axis=1).any(), \
        "no real corner should be (0, 0) — that was the bug"
    # All four real corners should be present, ignoring order.
    got = {tuple(c) for c in yz.astype(int)}
    assert got == set(real)


def test_corners_start_column_prepended():
    """The slice is along axis 0; the (start, y, z) layout is preserved."""
    row = _make_blob_row(desc=1, ident=10, start=42.5,
                         ncorners=3, corners_yz=[(1, 1), (1, 2), (2, 1)])
    gr = _build_graph_with_blob_rows([row])

    corners = gr.nodes[1]["corners"]
    assert corners.shape == (3, 3)
    np.testing.assert_array_equal(corners[:, 0], np.full(3, 42.5))


def test_corners_unaffected_when_ncorners_is_max():
    """Slicing to ncorners=12 must keep all 12 rows."""
    yz = [(i + 1, i + 1) for i in range(12)]
    row = _make_blob_row(desc=1, ident=10, start=0.0,
                         ncorners=12, corners_yz=yz)
    gr = _build_graph_with_blob_rows([row])

    corners = gr.nodes[1]["corners"]
    assert corners.shape == (12, 3)


def test_corners_per_blob_independent_truncation():
    """Two blobs with different ncorners are sliced independently."""
    row_a = _make_blob_row(desc=1, ident=10, start=0.0,
                           ncorners=3, corners_yz=[(1, 1), (1, 2), (2, 1)])
    row_b = _make_blob_row(desc=2, ident=20, start=10.0,
                           ncorners=7,
                           corners_yz=[(i, i + 1) for i in range(7)])
    gr = _build_graph_with_blob_rows([row_a, row_b])

    assert gr.nodes[1]["corners"].shape == (3, 3)
    assert gr.nodes[2]["corners"].shape == (7, 3)
