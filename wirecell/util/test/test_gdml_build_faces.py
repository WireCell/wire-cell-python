"""Tests for build_detector_faces() in wirecell.util.gdml."""

import xml.etree.ElementTree as ET
import pytest

from wirecell.util.gdml import (
    parse_define, parse_solids, parse_structure,
    extract_wires, build_detector_faces,
    FaceGeom, PlaneGeom, WireGeom,
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
def wires_world(ctx):
    return extract_wires(
        ctx["vol_tree"], ctx["defines"], ctx["solids"],
        root_vol="volWorld", patterns=_PATTERNS,
    )


@pytest.fixture
def faces_world(ctx, wires_world):
    return build_detector_faces(ctx["vol_tree"], wires_world, _PATTERNS)


# ── Return type ───────────────────────────────────────────────────────────────

def test_build_detector_faces_returns_list(faces_world):
    assert isinstance(faces_world, list)


def test_build_detector_faces_returns_facegeom_objects(faces_world):
    for f in faces_world:
        assert isinstance(f, FaceGeom)


# ── Face count ────────────────────────────────────────────────────────────────
# minimal.gdml: volTPC0 placed twice (volTPC0_top, volTPC0_bot) → 2 faces

def test_two_faces_from_world(faces_world):
    assert len(faces_world) == 2


def test_face_names_are_placement_names(faces_world):
    names = {f.name for f in faces_world}
    assert "volTPC0_top" in names
    assert "volTPC0_bot" in names


# ── Plane count per face ──────────────────────────────────────────────────────
# Each TPC has 3 planes (U, V, Z)

def test_each_face_has_three_planes(faces_world):
    for f in faces_world:
        assert len(f.planes) == 3, f"Face '{f.name}' has {len(f.planes)} planes"


def test_plane_objects_are_planegeom(faces_world):
    for f in faces_world:
        for p in f.planes:
            assert isinstance(p, PlaneGeom)


def test_plane_names_include_uvz(faces_world):
    for f in faces_world:
        pnames = {p.name for p in f.planes}
        assert any("PlaneU" in n for n in pnames), f"No U plane in {f.name}"
        assert any("PlaneV" in n for n in pnames), f"No V plane in {f.name}"
        assert any("PlaneZ" in n for n in pnames), f"No Z plane in {f.name}"


# ── Wire count per plane ──────────────────────────────────────────────────────
# Each plane has exactly 5 wires

def test_each_plane_has_five_wires(faces_world):
    for f in faces_world:
        for p in f.planes:
            assert len(p.wires) == 5, \
                f"Face '{f.name}', plane '{p.name}' has {len(p.wires)} wires"


def test_wire_objects_are_wiregeom(faces_world):
    for f in faces_world:
        for p in f.planes:
            for w in p.wires:
                assert isinstance(w, WireGeom)


# ── Total wire count preserved ────────────────────────────────────────────────

def test_total_wire_count(faces_world):
    total = sum(len(p.wires) for f in faces_world for p in f.planes)
    assert total == 30


# ── Faces are spatially distinct ─────────────────────────────────────────────
# Top and Bot placements are at different X coordinates

def test_faces_have_different_wire_positions(faces_world):
    face_by_name = {f.name: f for f in faces_world}
    top = face_by_name["volTPC0_top"]
    bot = face_by_name["volTPC0_bot"]
    # Pick U plane and first wire from each
    top_u = next(p for p in top.planes if "PlaneU" in p.name)
    bot_u = next(p for p in bot.planes if "PlaneU" in p.name)
    # Wire endpoints must differ between faces
    import numpy as np
    assert not np.allclose(top_u.wires[0].tail, bot_u.wires[0].tail), \
        "Top and Bot faces have identical wire positions — placement not tracked"


# ── face_name set on WireGeom ─────────────────────────────────────────────────

def test_all_wires_have_face_name(wires_world):
    assert all(w.face_name != "" for w in wires_world)


def test_wire_face_names_are_tpc_placement_names(wires_world):
    face_names = {w.face_name for w in wires_world}
    assert face_names == {"volTPC0_top", "volTPC0_bot"}


# ── Single-TPC extraction ─────────────────────────────────────────────────────

def test_faces_from_single_tpc(ctx):
    wires = extract_wires(
        ctx["vol_tree"], ctx["defines"], ctx["solids"],
        root_vol="volTPC0", patterns=_PATTERNS,
    )
    faces = build_detector_faces(ctx["vol_tree"], wires, _PATTERNS)
    # Starting from the LV directly (no face physvol traversed), face_name=""
    # → should produce 1 face group
    assert len(faces) == 1
    assert sum(len(p.wires) for f in faces for p in f.planes) == 15


# ── Empty input ───────────────────────────────────────────────────────────────

def test_empty_wires_returns_empty(ctx):
    faces = build_detector_faces(ctx["vol_tree"], [], _PATTERNS)
    assert faces == []
