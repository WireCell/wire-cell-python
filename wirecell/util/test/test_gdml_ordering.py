"""Tests for sort_wires_by_pitch and sort_planes_by_drift in wirecell.util.gdml."""

import math
import xml.etree.ElementTree as ET
import pytest
import numpy as np

from wirecell.util.gdml import (
    parse_define, parse_solids, parse_structure,
    extract_wires, build_detector_faces,
    sort_wires_by_pitch, sort_planes_by_drift,
    PlaneGeom, WireGeom,
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
def faces(gdml_root):
    defines  = parse_define(gdml_root)
    solids   = parse_solids(gdml_root)
    vol_tree = parse_structure(gdml_root, defines, solids)
    wires    = extract_wires(vol_tree, defines, solids, "volWorld", _PATTERNS)
    return build_detector_faces(vol_tree, wires, _PATTERNS)


@pytest.fixture
def top_face(faces):
    return next(f for f in faces if f.name == "volTPC0_top")


@pytest.fixture
def bot_face(faces):
    return next(f for f in faces if f.name == "volTPC0_bot")


# ── sort_wires_by_pitch: return type ─────────────────────────────────────────

def test_sort_wires_returns_list(top_face):
    plane = top_face.planes[0]
    result = sort_wires_by_pitch(plane)
    assert isinstance(result, list)


def test_sort_wires_preserves_length(top_face):
    for plane in top_face.planes:
        result = sort_wires_by_pitch(plane)
        assert len(result) == len(plane.wires)


def test_sort_wires_contains_all_wires(top_face):
    plane = top_face.planes[0]
    result = sort_wires_by_pitch(plane)
    assert set(id(w) for w in result) == set(id(w) for w in plane.wires)


def test_sort_wires_returns_wiregeom_objects(top_face):
    plane = top_face.planes[0]
    for w in sort_wires_by_pitch(plane):
        assert isinstance(w, WireGeom)


# ── sort_wires_by_pitch: correct order ───────────────────────────────────────
# Top face U-plane: wires spaced along Y at -9.52, -4.76, 0, 4.76, 9.52 mm.
# Sorted by pitch (ascending Y of midpoint) gives this exact sequence.

def test_sort_wires_U_ascending_y(top_face):
    u_plane = next(p for p in top_face.planes if "PlaneU" in p.name)
    sorted_wires = sort_wires_by_pitch(u_plane)
    mids_y = [0.5 * (w.head[1] + w.tail[1]) for w in sorted_wires]
    assert mids_y == sorted(mids_y), "U wires not sorted by ascending pitch (Y)"


def test_sort_wires_Z_ascending_z(top_face):
    # Z-plane wires: pitch direction is world Z; midpoints at z = -9.52..9.52
    z_plane = next(p for p in top_face.planes if "PlaneZ" in p.name)
    sorted_wires = sort_wires_by_pitch(z_plane)
    mids_z = [0.5 * (w.head[2] + w.tail[2]) for w in sorted_wires]
    assert mids_z == sorted(mids_z), "Z wires not sorted by ascending pitch (Z)"


def test_sort_wires_V_ascending_pitch(top_face):
    v_plane = next(p for p in top_face.planes if "PlaneV" in p.name)
    sorted_wires = sort_wires_by_pitch(v_plane)
    # For V wires (rWireV = Rx(+30°)), pitch direction has +Y component
    # Midpoints have the same Y spread so ascending Y is the sort key
    mids_y = [0.5 * (w.head[1] + w.tail[1]) for w in sorted_wires]
    assert mids_y == sorted(mids_y), "V wires not sorted by ascending pitch (Y)"


def test_sort_wires_stable_on_reversed_input(top_face):
    u_plane = next(p for p in top_face.planes if "PlaneU" in p.name)
    reversed_plane = PlaneGeom(name=u_plane.name, wires=list(reversed(u_plane.wires)))
    result = sort_wires_by_pitch(reversed_plane)
    expected = sort_wires_by_pitch(u_plane)
    assert [w.name for w in result] == [w.name for w in expected]


def test_sort_wires_single_wire():
    w = WireGeom(name="w0", tail=np.array([0., 0., -5.]),
                 head=np.array([0., 0., 5.]), radius=0.076, plane_name="P")
    plane = PlaneGeom(name="P", wires=[w])
    assert sort_wires_by_pitch(plane) == [w]


def test_sort_wires_empty_plane():
    plane = PlaneGeom(name="P", wires=[])
    assert sort_wires_by_pitch(plane) == []


# ── sort_planes_by_drift: return type ────────────────────────────────────────

def test_sort_planes_returns_list(top_face):
    assert isinstance(sort_planes_by_drift(top_face), list)


def test_sort_planes_preserves_length(top_face):
    result = sort_planes_by_drift(top_face)
    assert len(result) == len(top_face.planes)


def test_sort_planes_contains_all_planes(top_face):
    result = sort_planes_by_drift(top_face)
    assert set(id(p) for p in result) == set(id(p) for p in top_face.planes)


# ── sort_planes_by_drift: Top face ───────────────────────────────────────────
# Top TPC planes at world X: U≈10.24, V≈15.0, Z≈19.76.
# Drift indicator cross(U_dir, V_dir) = +X → sort ascending X → U, V, Z.

def test_sort_planes_top_first_is_U(top_face):
    sorted_planes = sort_planes_by_drift(top_face)
    assert "PlaneU" in sorted_planes[0].name


def test_sort_planes_top_last_is_Z(top_face):
    sorted_planes = sort_planes_by_drift(top_face)
    assert "PlaneZ" in sorted_planes[-1].name


def test_sort_planes_top_middle_is_V(top_face):
    sorted_planes = sort_planes_by_drift(top_face)
    assert "PlaneV" in sorted_planes[1].name


def test_sort_planes_top_ascending_x(top_face):
    sorted_planes = sort_planes_by_drift(top_face)
    xs = [np.mean([0.5*(w.head[0]+w.tail[0]) for w in p.wires]) for p in sorted_planes]
    assert xs == sorted(xs), f"Top face planes not in ascending X order: {xs}"


# ── sort_planes_by_drift: Bot face ───────────────────────────────────────────
# Bot TPC planes at world X: U≈-14.24, V≈-19.0, Z≈-23.76.
# Drift indicator cross(U_bot, V_bot) = -X → sort ascending (-X projection) → U, V, Z.

def test_sort_planes_bot_first_is_U(bot_face):
    sorted_planes = sort_planes_by_drift(bot_face)
    assert "PlaneU" in sorted_planes[0].name


def test_sort_planes_bot_last_is_Z(bot_face):
    sorted_planes = sort_planes_by_drift(bot_face)
    assert "PlaneZ" in sorted_planes[-1].name


def test_sort_planes_bot_first_is_highest_x(bot_face):
    # U is closest to cathode = highest X among Bot planes (least negative)
    sorted_planes = sort_planes_by_drift(bot_face)
    first_x = np.mean([0.5*(w.head[0]+w.tail[0]) for w in sorted_planes[0].wires])
    for p in sorted_planes[1:]:
        px = np.mean([0.5*(w.head[0]+w.tail[0]) for w in p.wires])
        assert first_x > px, f"Bot face first plane x={first_x} not > {px}"


# ── sort_planes_by_drift: edge cases ─────────────────────────────────────────

def test_sort_planes_single_plane(top_face):
    single = type(top_face)(name="f", planes=[top_face.planes[0]])
    result = sort_planes_by_drift(single)
    assert len(result) == 1
