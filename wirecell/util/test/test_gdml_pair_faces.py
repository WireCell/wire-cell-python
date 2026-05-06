"""Tests for pair_faces_into_anodes() in wirecell.util.gdml."""

import xml.etree.ElementTree as ET
import pytest
import numpy as np

from wirecell.util.gdml import (
    parse_define, parse_solids, parse_structure,
    extract_wires, build_detector_faces,
    pair_faces_into_anodes,
    AnodeGeom, FaceGeom,
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


# ── Return type ───────────────────────────────────────────────────────────────

def test_pair_faces_returns_list(faces):
    result = pair_faces_into_anodes(faces, "vd")
    assert isinstance(result, list)


def test_pair_faces_returns_anodegeom_objects(faces):
    result = pair_faces_into_anodes(faces, "vd")
    for a in result:
        assert isinstance(a, AnodeGeom)


# ── VD pairing: minimal GDML has 2 faces → 1 anode ──────────────────────────

def test_vd_two_faces_give_one_anode(faces):
    anodes = pair_faces_into_anodes(faces, "vd")
    assert len(anodes) == 1


def test_vd_anode_has_two_faces(faces):
    anodes = pair_faces_into_anodes(faces, "vd")
    assert len(anodes[0].faces) == 2


def test_vd_anode_faces_are_facegeom(faces):
    anodes = pair_faces_into_anodes(faces, "vd")
    for f in anodes[0].faces:
        assert isinstance(f, FaceGeom)


def test_vd_anode_contains_both_placement_faces(faces):
    anodes = pair_faces_into_anodes(faces, "vd")
    face_names = {f.name for f in anodes[0].faces}
    assert "volTPC0_top" in face_names
    assert "volTPC0_bot" in face_names


# ── VD pairing: open-book rule — same TPC LV, side-by-side in Y-Z ────────────
# The two faces share the same drift-axis X position cluster, i.e., their
# wire-plane stacks overlap in Y-Z but face in opposite X directions.

def test_vd_paired_faces_have_same_tpc_lv(faces):
    # Both faces come from volTPC0 (same LV placed twice); face_name encodes
    # "volTPC0_top" and "volTPC0_bot".  The TPC LV name is the part before the
    # last underscore-prefixed token.
    anodes = pair_faces_into_anodes(faces, "vd")
    face_names = [f.name for f in anodes[0].faces]
    # Both should start with the same LV prefix "volTPC0"
    lv_names = {"_".join(n.split("_")[:-1]) for n in face_names}
    assert len(lv_names) == 1, f"Faces from different LVs paired: {face_names}"


# ── VD pairing: opposite drift directions ────────────────────────────────────
# Top face planes are at positive X; Bot face planes are at negative X.
# In a paired anode the two face-mean X positions should be on opposite sides
# of zero (or at minimum, substantially different from each other).

def test_vd_paired_faces_have_opposite_drift_sides(faces):
    anodes = pair_faces_into_anodes(faces, "vd")
    f0, f1 = anodes[0].faces
    x0 = np.mean([0.5*(w.head[0]+w.tail[0]) for p in f0.planes for w in p.wires])
    x1 = np.mean([0.5*(w.head[0]+w.tail[0]) for p in f1.planes for w in p.wires])
    # One should be positive, one negative
    assert x0 * x1 < 0, f"Paired faces both on same side of cathode: x={x0:.2f}, {x1:.2f}"


# ── VD pairing: odd number of faces raises ───────────────────────────────────

def test_vd_odd_faces_raises(faces):
    # Add a third orphan face
    orphan = FaceGeom(name="orphan", planes=[])
    with pytest.raises((ValueError, AssertionError)):
        pair_faces_into_anodes(faces + [orphan], "vd")


# ── HD pairing: synthetic closed-book pair ───────────────────────────────────
# Create two synthetic faces with coincident Y-Z extents and opposite X sides.

def _make_synthetic_face(name, x_center, n_wires=3):
    """Create a minimal FaceGeom with parallel wires at given X."""
    wires = []
    for i in range(n_wires):
        y = float(i * 5.0)
        w = type('WireGeom', (), {
            'name': f'{name}_w{i}',
            'tail': np.array([x_center, y, -10.0]),
            'head': np.array([x_center, y,  10.0]),
            'radius': 0.076,
            'plane_name': f'{name}_plane',
            'face_name': name,
            'channel': None, 'segment': None,
        })()
        wires.append(w)

    from wirecell.util.gdml import PlaneGeom, FaceGeom
    plane = PlaneGeom(name=f'{name}_plane', wires=wires)
    face = FaceGeom(name=name, planes=[plane])
    return face


def test_hd_two_faces_give_one_anode():
    # HD closed-book: two faces at x=+5 and x=-5 with same Y-Z footprint
    face_a = _make_synthetic_face("face_a", x_center=5.0)
    face_b = _make_synthetic_face("face_b", x_center=-5.0)
    anodes = pair_faces_into_anodes([face_a, face_b], "hd")
    assert len(anodes) == 1


def test_hd_anode_has_two_faces():
    face_a = _make_synthetic_face("face_a", x_center=5.0)
    face_b = _make_synthetic_face("face_b", x_center=-5.0)
    anodes = pair_faces_into_anodes([face_a, face_b], "hd")
    assert len(anodes[0].faces) == 2


def test_hd_odd_faces_raises():
    face_a = _make_synthetic_face("face_a", x_center=5.0)
    with pytest.raises((ValueError, AssertionError)):
        pair_faces_into_anodes([face_a], "hd")


# ── Unknown connectivity mode ─────────────────────────────────────────────────

def test_unknown_mode_raises(faces):
    with pytest.raises((ValueError, KeyError, NotImplementedError)):
        pair_faces_into_anodes(faces, "unknown_mode")


# ── Empty input ───────────────────────────────────────────────────────────────

def test_empty_faces_returns_empty():
    assert pair_faces_into_anodes([], "vd") == []
