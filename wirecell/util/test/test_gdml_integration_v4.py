"""Integration test: protodunevd_v4 gdml-to-wires vs known-good wires file.

Marked with pytest.mark.integration so it can be skipped in fast CI runs:

    pytest -m "not integration"   # fast suite (skips this file)
    pytest -m integration         # run only integration tests

Both the GDML source and the reference wires file live outside this package:

    GDML:  ../dunecore/dunecore/Geometry/gdml/protodunevd_v4_refactored.gdml
    REF:   ../dunereco/dunereco/DUNEWireCell/protodunevd/protodunevd-wires-larsoft-v3.json.bz2

Tests are skipped automatically when either file is absent.
"""

import pathlib
import numpy as np
import pytest

from wirecell.util.gdml import load_config, convert
from wirecell.util.wires import persist

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE = pathlib.Path(__file__).parent
_GDML = _HERE.parents[3] / "dunecore/dunecore/Geometry/gdml/protodunevd_v4_refactored.gdml"
_REF  = _HERE.parents[3] / "dunereco/dunereco/DUNEWireCell/protodunevd/protodunevd-wires-larsoft-v3.json.bz2"

pytestmark = pytest.mark.integration


def _skip_if_missing():
    if not _GDML.exists():
        pytest.skip(f"GDML not found: {_GDML}")
    if not _REF.exists():
        pytest.skip(f"Reference wires file not found: {_REF}")


@pytest.fixture(scope="module")
def stores():
    _skip_if_missing()
    cfg = load_config("protodunevd_v4")
    ours = convert(_GDML, cfg)
    ref  = persist.load(str(_REF))
    return ours, ref


# ── Top-level counts ──────────────────────────────────────────────────────────

def test_anode_count(stores):
    ours, ref = stores
    assert len(ours.anodes) == len(ref.anodes) == 8


def test_face_count(stores):
    ours, ref = stores
    assert len(ours.faces) == len(ref.faces) == 16


def test_plane_count(stores):
    ours, ref = stores
    assert len(ours.planes) == len(ref.planes) == 48


def test_total_wire_count(stores):
    ours, ref = stores
    assert len(ours.wires) == len(ref.wires) == 13840


def test_has_detector(stores):
    ours, _ = stores
    assert len(ours.detectors) == 1
    assert ours.detectors[0].ident == 0


# ── Per-anode structure ───────────────────────────────────────────────────────

def test_each_anode_has_two_faces(stores):
    ours, _ = stores
    for ai, anode in enumerate(ours.anodes):
        assert len(anode.faces) == 2, f"anode {ai} has {len(anode.faces)} faces"


def test_each_face_has_three_planes(stores):
    ours, _ = stores
    for ai, anode in enumerate(ours.anodes):
        for fi in anode.faces:
            face = ours.faces[fi]
            assert len(face.planes) == 3, \
                f"anode {ai} face {fi} has {len(face.planes)} planes"


def test_plane_idents_are_0_1_2(stores):
    ours, _ = stores
    for face in ours.faces:
        idents = sorted(ours.planes[pi].ident for pi in face.planes)
        assert idents == [0, 1, 2]


# ── Wire count per plane matches reference ────────────────────────────────────
# The U/V planes of the positive-X (seg=1) face can swap relative to the
# reference due to different drift-order conventions, so we only assert the
# Z-plane (ident=2) count and the total per face.

def test_z_wire_count_per_face(stores):
    ours, ref = stores
    for ai in range(len(ours.anodes)):
        oa, ra = ours.anodes[ai], ref.anodes[ai]
        for fi_idx in range(2):
            ofi = oa.faces[fi_idx]
            rfi = ra.faces[fi_idx]
            oz = next(ours.planes[pi] for pi in ours.faces[ofi].planes
                      if ours.planes[pi].ident == 2)
            rz = next(ref.planes[pi] for pi in ref.faces[rfi].planes
                      if ref.planes[pi].ident == 2)
            assert len(oz.wires) == len(rz.wires), \
                f"anode {ai} face {fi_idx} Z-plane: ours={len(oz.wires)} ref={len(rz.wires)}"


def test_total_wires_per_face_matches_reference(stores):
    ours, ref = stores
    for ai in range(len(ours.anodes)):
        oa, ra = ours.anodes[ai], ref.anodes[ai]
        for fi_idx in range(2):
            ofi = oa.faces[fi_idx]
            rfi = ra.faces[fi_idx]
            o_total = sum(len(ours.planes[pi].wires) for pi in ours.faces[ofi].planes)
            r_total = sum(len(ref.planes[pi].wires) for pi in ref.faces[rfi].planes)
            assert o_total == r_total, \
                f"anode {ai} face {fi_idx}: ours={o_total} ref={r_total} total wires"


# ── Wire endpoint coordinates match reference ─────────────────────────────────
# Sample the Z plane of each face (exact wire-count match guaranteed above).
# Tolerance: 0.1 mm (accommodates floating-point rounding in transforms).

_COORD_TOL_MM = 0.1


def _z_plane_coords(store, anode_idx, face_idx):
    """Return sorted list of (tail_xyz, head_xyz) tuples for Z-plane wires."""
    anode = store.anodes[anode_idx]
    face  = store.faces[anode.faces[face_idx]]
    zplane = next(store.planes[pi] for pi in face.planes
                  if store.planes[pi].ident == 2)
    pts = []
    for wi in zplane.wires:
        w = store.wires[wi]
        t = store.points[w.tail]
        h = store.points[w.head]
        pts.append(((t.x, t.y, t.z), (h.x, h.y, h.z)))
    return sorted(pts)


@pytest.mark.parametrize("anode_idx,face_idx", [
    (0, 0), (0, 1),
    (3, 0), (3, 1),
    (7, 0), (7, 1),
])
def test_z_plane_wire_coords(stores, anode_idx, face_idx):
    ours, ref = stores
    our_pts = _z_plane_coords(ours, anode_idx, face_idx)
    ref_pts = _z_plane_coords(ref,  anode_idx, face_idx)
    assert len(our_pts) == len(ref_pts)
    for i, (op, rp) in enumerate(zip(our_pts, ref_pts)):
        dt = np.linalg.norm(np.array(op[0]) - np.array(rp[0]))
        dh = np.linalg.norm(np.array(op[1]) - np.array(rp[1]))
        assert dt < _COORD_TOL_MM, \
            f"anode{anode_idx} face{face_idx} wire{i} tail dist={dt:.4f}mm"
        assert dh < _COORD_TOL_MM, \
            f"anode{anode_idx} face{face_idx} wire{i} head dist={dh:.4f}mm"


# ── Channel uniqueness ────────────────────────────────────────────────────────

def test_channels_unique_per_anode(stores):
    ours, _ = stores
    for ai, anode in enumerate(ours.anodes):
        channels = set()
        for fi in anode.faces:
            for pi in ours.faces[fi].planes:
                for wi in ours.planes[pi].wires:
                    channels.add(ours.wires[wi].channel)
        assert len(channels) == 1536, \
            f"anode {ai}: expected 1536 unique channels, got {len(channels)}"


def test_channels_globally_unique(stores):
    ours, _ = stores
    anode_channels = []
    for anode in ours.anodes:
        ch = {ours.wires[wi].channel
              for fi in anode.faces
              for pi in ours.faces[fi].planes
              for wi in ours.planes[pi].wires}
        anode_channels.append(ch)
    for i in range(len(anode_channels)):
        for j in range(i + 1, len(anode_channels)):
            assert anode_channels[i].isdisjoint(anode_channels[j]), \
                f"anodes {i} and {j} share channel IDs"


# ── All indices in-range ──────────────────────────────────────────────────────

def test_all_indices_in_range(stores):
    ours, _ = stores
    for det in ours.detectors:
        for ai in det.anodes:
            assert 0 <= ai < len(ours.anodes)
    for anode in ours.anodes:
        for fi in anode.faces:
            assert 0 <= fi < len(ours.faces)
            for pi in ours.faces[fi].planes:
                assert 0 <= pi < len(ours.planes)
                for wi in ours.planes[pi].wires:
                    assert 0 <= wi < len(ours.wires)
                    w = ours.wires[wi]
                    assert 0 <= w.tail < len(ours.points)
                    assert 0 <= w.head < len(ours.points)
