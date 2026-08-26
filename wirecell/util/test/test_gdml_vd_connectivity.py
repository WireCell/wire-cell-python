"""Tests for find_vd_connected_pairs() in wirecell.util.gdml."""

import pytest
import numpy as np

from wirecell.util.gdml import (
    find_vd_connected_pairs,
    WireGeom, PlaneGeom, FaceGeom, AnodeGeom,
)

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic VD anode fixture
#
# Two induction planes (U, V) with cross-face jumpers and one collection
# plane (Z) without.  Each induction wire in face0 shares one endpoint with
# the corresponding wire in face1 (the shared z=0 boundary).
#
# Convention (from wcpy-zv1 research):
#   • face0 (index 0) → segment = 1   (further from electronics)
#   • face1 (index 1) → segment = 0   (directly connected to electronics)
#   • Collection (Z) wires → segment = 0, connected_to = None
# ─────────────────────────────────────────────────────────────────────────────

def _w(name, tail, head, plane_name, face_name=""):
    return WireGeom(
        name=name,
        tail=np.array(tail, dtype=float),
        head=np.array(head, dtype=float),
        radius=0.076,
        plane_name=plane_name,
        face_name=face_name,
    )


def _make_vd_anode():
    """
    face0  (seg=1 for connected wires)
      planeU: U0_f0 tail=(0,-5,-50) head=(0,-5,0)   ← head at z=0
              U1_f0 tail=(0, 0,-50) head=(0, 0,0)
      planeZ: Z0_f0 tail=(0,-5,-25) head=(0,-5,25)  ← no cross-face match
              Z1_f0 tail=(0, 0,-25) head=(0, 0,25)

    face1  (seg=0 for connected wires)
      planeU: U0_f1 tail=(0,-5, 0)  head=(0,-5,50)  ← tail at z=0
              U1_f1 tail=(0, 0, 0)  head=(0, 0,50)
      planeZ: Z0_f1 tail=(0,-5,-25) head=(0,-5,25)  ← same positions, no match
              Z1_f1 tail=(0, 0,-25) head=(0, 0,25)
    """
    # face0
    u0f0 = _w("U0_f0", [0,-5,-50], [0,-5, 0], "planeU", "face0")
    u1f0 = _w("U1_f0", [0, 0,-50], [0, 0, 0], "planeU", "face0")
    z0f0 = _w("Z0_f0", [0,-5,-25], [0,-5,25], "planeZ", "face0")
    z1f0 = _w("Z1_f0", [0, 0,-25], [0, 0,25], "planeZ", "face0")
    planeU_f0 = PlaneGeom(name="planeU", wires=[u0f0, u1f0])
    planeZ_f0 = PlaneGeom(name="planeZ", wires=[z0f0, z1f0])
    face0 = FaceGeom(name="face0", planes=[planeU_f0, planeZ_f0])

    # face1
    u0f1 = _w("U0_f1", [0,-5,  0], [0,-5,50], "planeU", "face1")
    u1f1 = _w("U1_f1", [0, 0,  0], [0, 0,50], "planeU", "face1")
    z0f1 = _w("Z0_f1", [0,-5,-25], [0,-5,25], "planeZ", "face1")
    z1f1 = _w("Z1_f1", [0, 0,-25], [0, 0,25], "planeZ", "face1")
    planeU_f1 = PlaneGeom(name="planeU", wires=[u0f1, u1f1])
    planeZ_f1 = PlaneGeom(name="planeZ", wires=[z0f1, z1f1])
    face1 = FaceGeom(name="face1", planes=[planeU_f1, planeZ_f1])

    return AnodeGeom(faces=[face0, face1])


@pytest.fixture
def anode():
    return _make_vd_anode()


@pytest.fixture
def result(anode):
    return find_vd_connected_pairs(anode, nearness_tolerance=1.0)


# ── Return type ───────────────────────────────────────────────────────────────

def test_returns_dict(result):
    assert isinstance(result, dict)


def test_all_wires_have_entries(anode, result):
    all_names = {w.name for f in anode.faces for p in f.planes for w in p.wires}
    assert set(result.keys()) == all_names


def test_each_entry_has_segment(result):
    for v in result.values():
        assert "segment" in v


def test_each_entry_has_connected_to(result):
    for v in result.values():
        assert "connected_to" in v


# ── Induction wire pairing ────────────────────────────────────────────────────

def test_u0_f0_is_connected(result):
    assert result["U0_f0"]["connected_to"] == "U0_f1"


def test_u0_f1_is_connected(result):
    assert result["U0_f1"]["connected_to"] == "U0_f0"


def test_u1_pair_is_connected(result):
    assert result["U1_f0"]["connected_to"] == "U1_f1"
    assert result["U1_f1"]["connected_to"] == "U1_f0"


# ── Segment polarity ──────────────────────────────────────────────────────────
# face0 (index 0) → segment=1; face1 (index 1) → segment=0

def test_face0_induction_segment_is_1(result):
    assert result["U0_f0"]["segment"] == 1
    assert result["U1_f0"]["segment"] == 1


def test_face1_induction_segment_is_0(result):
    assert result["U0_f1"]["segment"] == 0
    assert result["U1_f1"]["segment"] == 0


# ── Collection (Z) wires are standalone ──────────────────────────────────────

def test_collection_wires_have_no_partner(result):
    for name in ("Z0_f0", "Z1_f0", "Z0_f1", "Z1_f1"):
        assert result[name]["connected_to"] is None, \
            f"{name} should not be connected"


def test_collection_wires_have_segment_0(result):
    for name in ("Z0_f0", "Z1_f0", "Z0_f1", "Z1_f1"):
        assert result[name]["segment"] == 0


# ── Tolerance boundary ────────────────────────────────────────────────────────

def test_zero_tolerance_finds_exact_matches():
    """Exact coincident endpoints (distance=0) are matched at any tolerance."""
    anode = _make_vd_anode()
    result = find_vd_connected_pairs(anode, nearness_tolerance=0.0)
    assert result["U0_f0"]["connected_to"] == "U0_f1"


def test_tolerance_too_small_misses_nearby_endpoints():
    """Endpoints that are 2 mm apart are NOT matched with 1 mm tolerance."""
    u0f0 = _w("U0_f0", [0,-5,-50], [0,-5, 0], "planeU", "face0")
    u0f1 = _w("U0_f1", [0,-5, 2],  [0,-5,52], "planeU", "face1")  # gap = 2 mm
    f0 = FaceGeom(name="face0", planes=[PlaneGeom("planeU", [u0f0])])
    f1 = FaceGeom(name="face1", planes=[PlaneGeom("planeU", [u0f1])])
    anode = AnodeGeom(faces=[f0, f1])
    result = find_vd_connected_pairs(anode, nearness_tolerance=1.0)
    assert result["U0_f0"]["connected_to"] is None
    assert result["U0_f1"]["connected_to"] is None


def test_tolerance_large_enough_finds_nearby_endpoints():
    """Endpoints that are 2 mm apart ARE matched with 3 mm tolerance."""
    u0f0 = _w("U0_f0", [0,-5,-50], [0,-5, 0], "planeU", "face0")
    u0f1 = _w("U0_f1", [0,-5, 2],  [0,-5,52], "planeU", "face1")  # gap = 2 mm
    f0 = FaceGeom(name="face0", planes=[PlaneGeom("planeU", [u0f0])])
    f1 = FaceGeom(name="face1", planes=[PlaneGeom("planeU", [u0f1])])
    anode = AnodeGeom(faces=[f0, f1])
    result = find_vd_connected_pairs(anode, nearness_tolerance=3.0)
    assert result["U0_f0"]["connected_to"] == "U0_f1"


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_wrong_face_count_raises():
    face = FaceGeom(name="solo", planes=[])
    with pytest.raises((ValueError, AssertionError)):
        find_vd_connected_pairs(AnodeGeom(faces=[face]), nearness_tolerance=1.0)


def test_three_planes_uvz_only_uv_connected():
    """Add a V induction plane; both U and V should be matched, Z not."""
    u0f0 = _w("U0_f0", [0,-5,-50], [0,-5, 0], "planeU", "face0")
    v0f0 = _w("V0_f0", [0,-5,-50], [0,-5, 0], "planeV", "face0")
    z0f0 = _w("Z0_f0", [0,-5,-25], [0,-5,25], "planeZ", "face0")
    u0f1 = _w("U0_f1", [0,-5,  0], [0,-5,50], "planeU", "face1")
    v0f1 = _w("V0_f1", [0,-5,  0], [0,-5,50], "planeV", "face1")
    z0f1 = _w("Z0_f1", [0,-5,-25], [0,-5,25], "planeZ", "face1")

    f0 = FaceGeom("face0", [PlaneGeom("planeU",[u0f0]), PlaneGeom("planeV",[v0f0]), PlaneGeom("planeZ",[z0f0])])
    f1 = FaceGeom("face1", [PlaneGeom("planeU",[u0f1]), PlaneGeom("planeV",[v0f1]), PlaneGeom("planeZ",[z0f1])])
    anode = AnodeGeom(faces=[f0, f1])
    result = find_vd_connected_pairs(anode, nearness_tolerance=1.0)

    assert result["U0_f0"]["connected_to"] == "U0_f1"
    assert result["V0_f0"]["connected_to"] == "V0_f1"
    assert result["Z0_f0"]["connected_to"] is None
