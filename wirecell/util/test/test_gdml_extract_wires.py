"""Tests for extract_wires() in wirecell.util.gdml."""

import math
import xml.etree.ElementTree as ET
import pytest
import numpy as np

from wirecell.util.gdml import (
    parse_define, parse_solids, parse_structure,
    extract_wires, WireGeom,
)

_PATTERNS = {
    "wire":  r"volTPCWire[UVZ]\d+",
    "plane": r"volTPCPlane[UVZ]\d+",
    "face":  r"volTPC\d+",
}


@pytest.fixture
def gdml_root(minimal_gdml_path):
    return ET.parse(minimal_gdml_path).getroot()


@pytest.fixture
def ctx(gdml_root):
    defines  = parse_define(gdml_root)
    solids   = parse_solids(gdml_root)
    vol_tree = parse_structure(gdml_root, defines, solids)
    return {"defines": defines, "solids": solids, "vol_tree": vol_tree}


@pytest.fixture
def wires_from_world(ctx):
    return extract_wires(
        ctx["vol_tree"], ctx["defines"], ctx["solids"],
        root_vol="volWorld", patterns=_PATTERNS,
    )


# ── Return type ───────────────────────────────────────────────────────────────

def test_extract_wires_returns_list(wires_from_world):
    assert isinstance(wires_from_world, list)


def test_extract_wires_returns_wiregeom_objects(wires_from_world):
    for w in wires_from_world:
        assert isinstance(w, WireGeom)


# ── Wire count ────────────────────────────────────────────────────────────────
# minimal.gdml: 5 U + 5 V + 5 Z wires per TPC placement × 2 placements = 30

def test_extract_wires_total_count(wires_from_world):
    assert len(wires_from_world) == 30


def test_extract_wires_from_tpc_level(ctx):
    # Starting from a single TPC should yield half the wires (15 = 5 U + 5 V + 5 Z)
    wires = extract_wires(
        ctx["vol_tree"], ctx["defines"], ctx["solids"],
        root_vol="volTPC0", patterns=_PATTERNS,
    )
    assert len(wires) == 15


# ── Endpoints ─────────────────────────────────────────────────────────────────

def test_all_wires_have_non_none_tail(wires_from_world):
    assert all(w.tail is not None for w in wires_from_world)


def test_all_wires_have_non_none_head(wires_from_world):
    assert all(w.head is not None for w in wires_from_world)


def test_tail_and_head_are_ndarrays(wires_from_world):
    w = wires_from_world[0]
    assert isinstance(w.tail, np.ndarray) and w.tail.shape == (3,)
    assert isinstance(w.head, np.ndarray) and w.head.shape == (3,)


def test_tail_and_head_differ(wires_from_world):
    for w in wires_from_world:
        assert not np.allclose(w.tail, w.head), f"tail == head for {w.name}"


# ── Wire length ───────────────────────────────────────────────────────────────

def test_U_wire_length(wires_from_world):
    # CRMWireU: full length 100 mm → distance between endpoints = 100 mm
    u_wires = [w for w in wires_from_world if "WireU" in w.name]
    for w in u_wires:
        length = np.linalg.norm(w.head - w.tail)
        assert abs(length - 100.0) < 1e-6, f"U wire length {length} != 100"


def test_Z_wire_length(wires_from_world):
    # CRMWireZ: full length 50 mm
    z_wires = [w for w in wires_from_world if "WireZ" in w.name]
    for w in z_wires:
        length = np.linalg.norm(w.head - w.tail)
        assert abs(length - 50.0) < 1e-6, f"Z wire length {length} != 50"


# ── Specific endpoint values (Top placement volTPC0_top) ─────────────────────
#
# Transform chain for volTPCWireU0 in the Top TPC:
#   T_top   : pos=[15.0, 0, 0],   rot=[0, 0, 0]         → R=I, t=[15,0,0]
#   T_planeU: pos=[-4.76, 0, 0],  rot=[0, 0, 0]         → R=I, t=[-4.76,0,0]
#   T_wireU0: pos=[0, -9.52, 0],  rot=[30°, 0, 0]       → R=Rx(-30°), t=[0,-9.52,0]
#
# Wire center in world: (15-4.76+0, 0+0-9.52, 0) = (10.24, -9.52, 0)
# Rx(-30°) @ [0, 0, ±50]:  y' = ∓50·sin(-30°) = ±25,  z' = ±50·cos(-30°) = ±25√3
# head = (10.24, -9.52+25,  25√3)  ≈ (10.24, 15.48,  43.301)
# tail = (10.24, -9.52-25, -25√3)  ≈ (10.24, -34.52, -43.301)

def _expected_wireU0_top():
    cx, cy = 15.0 - 4.76, -9.52
    half_z = 50.0
    theta = math.radians(-30)  # active rotation for rWireU
    dy = -half_z * math.sin(theta)
    dz =  half_z * math.cos(theta)
    head = np.array([cx, cy + dy, dz])
    tail = np.array([cx, cy - dy, -dz])
    return head, tail


def test_wireU0_top_head(wires_from_world):
    # Find the U0 wire that belongs to the Top TPC (positive x world position)
    u0_wires = [w for w in wires_from_world if w.name == "volTPCWireU0"]
    top = max(u0_wires, key=lambda w: w.head[0])  # Top TPC is at higher x
    expected_head, _ = _expected_wireU0_top()
    np.testing.assert_allclose(top.head, expected_head, atol=1e-6)


def test_wireU0_top_tail(wires_from_world):
    u0_wires = [w for w in wires_from_world if w.name == "volTPCWireU0"]
    top = max(u0_wires, key=lambda w: w.head[0])
    _, expected_tail = _expected_wireU0_top()
    np.testing.assert_allclose(top.tail, expected_tail, atol=1e-6)


# ── Z-wire axis direction ─────────────────────────────────────────────────────
# rWireW = x=90°, active R = Rx(-90°).
# Rx(-90°) @ [0, 0, ±25]:  y' = ±25,  z' = 0
# So Z wires should point along world Y, with zero Z displacement.

def test_Z_wires_point_along_Y(wires_from_world):
    z_wires = [w for w in wires_from_world if "WireZ" in w.name]
    for w in z_wires:
        axis = w.head - w.tail
        # x and z components should be zero; only Y non-zero
        assert abs(axis[0]) < 1e-6, f"Z wire {w.name} has non-zero x component {axis[0]}"
        assert abs(axis[2]) < 1e-6, f"Z wire {w.name} has non-zero z component {axis[2]}"
        assert abs(axis[1]) > 1.0,  f"Z wire {w.name} has near-zero Y component"


# ── Radius ────────────────────────────────────────────────────────────────────

def test_all_wires_have_positive_radius(wires_from_world):
    assert all(w.radius > 0 for w in wires_from_world)


def test_wire_radius_value(wires_from_world):
    # All wire solids in minimal.gdml have rmax=0.076 mm
    for w in wires_from_world:
        assert abs(w.radius - 0.076) < 1e-9, f"{w.name} radius {w.radius} != 0.076"


# ── Plane name ────────────────────────────────────────────────────────────────

def test_all_wires_have_plane_name(wires_from_world):
    assert all(w.plane_name != "" for w in wires_from_world)


def test_U_wires_have_U_plane_name(wires_from_world):
    u_wires = [w for w in wires_from_world if "WireU" in w.name]
    for w in u_wires:
        assert "PlaneU" in w.plane_name, f"{w.name} plane_name={w.plane_name!r}"


def test_Z_wires_have_Z_plane_name(wires_from_world):
    z_wires = [w for w in wires_from_world if "WireZ" in w.name]
    for w in z_wires:
        assert "PlaneZ" in w.plane_name, f"{w.name} plane_name={w.plane_name!r}"


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_unknown_root_vol_returns_empty(ctx):
    wires = extract_wires(
        ctx["vol_tree"], ctx["defines"], ctx["solids"],
        root_vol="volDoesNotExist", patterns=_PATTERNS,
    )
    assert wires == []
