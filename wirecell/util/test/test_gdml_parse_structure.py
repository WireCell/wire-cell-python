"""Tests for parse_structure() in wirecell.util.gdml."""

import math
import xml.etree.ElementTree as ET
import pytest
import numpy as np

from wirecell.util.gdml import parse_define, parse_solids, parse_structure


@pytest.fixture
def gdml_root(minimal_gdml_path):
    return ET.parse(minimal_gdml_path).getroot()


@pytest.fixture
def parsed(gdml_root):
    defines = parse_define(gdml_root)
    solids = parse_solids(gdml_root)
    return parse_structure(gdml_root, defines, solids)


# ── Return structure ──────────────────────────────────────────────────────────

def test_parse_structure_returns_dict(parsed):
    assert isinstance(parsed, dict)


def test_known_volumes_are_keys(parsed):
    for name in ("volTPC0", "volTPCPlaneU0", "volTPCPlaneV0", "volTPCPlaneZ0", "volWorld"):
        assert name in parsed, f"Expected '{name}' in structure"


def test_wire_volumes_are_keys(parsed):
    assert "volTPCWireU0" in parsed
    assert "volTPCWireZ4" in parsed


def test_each_entry_has_solid(parsed):
    for name, entry in parsed.items():
        assert "solid" in entry, f"'{name}' missing 'solid'"


def test_each_entry_has_physvols(parsed):
    for name, entry in parsed.items():
        assert "physvols" in entry, f"'{name}' missing 'physvols'"


# ── Solid references ──────────────────────────────────────────────────────────

def test_tpc_solid_ref(parsed):
    assert parsed["volTPC0"]["solid"] == "CRM"


def test_plane_solid_ref(parsed):
    assert parsed["volTPCPlaneU0"]["solid"] == "CRMPlaneU"


def test_wire_solid_ref(parsed):
    assert parsed["volTPCWireU0"]["solid"] == "CRMWireU"


# ── Physvol counts ────────────────────────────────────────────────────────────

def test_volTPC0_has_four_physvols(parsed):
    assert len(parsed["volTPC0"]["physvols"]) == 4


def test_volTPCPlaneU0_has_five_physvols(parsed):
    assert len(parsed["volTPCPlaneU0"]["physvols"]) == 5


def test_wire_volume_has_no_physvols(parsed):
    # Wire logical volumes are leaf nodes
    assert len(parsed["volTPCWireU0"]["physvols"]) == 0


def test_volWorld_has_two_physvols(parsed):
    assert len(parsed["volWorld"]["physvols"]) == 2


# ── Position resolution via positionref ──────────────────────────────────────

def test_planeU0_placement_position(parsed):
    # volTPCPlaneU0_pv in volTPC0: positionref=posPlaneU0 → [-4.76, 0, 0]
    pvs = parsed["volTPC0"]["physvols"]
    pv = next(p for p in pvs if p["vol"] == "volTPCPlaneU0")
    np.testing.assert_allclose(pv["pos"], [-4.76, 0.0, 0.0], atol=1e-9)


def test_planeZ0_placement_position(parsed):
    # positionref=posPlaneZ0 → [4.76, 0, 0]
    pvs = parsed["volTPC0"]["physvols"]
    pv = next(p for p in pvs if p["vol"] == "volTPCPlaneZ0")
    np.testing.assert_allclose(pv["pos"], [4.76, 0.0, 0.0], atol=1e-9)


def test_wireU0_placement_position(parsed):
    # positionref=posWireU0 → [0, -9.52, 0]
    pvs = parsed["volTPCPlaneU0"]["physvols"]
    pv = next(p for p in pvs if p["vol"] == "volTPCWireU0")
    np.testing.assert_allclose(pv["pos"], [0.0, -9.52, 0.0], atol=1e-9)


# ── Position resolution via inline <position> ────────────────────────────────

def test_active_inline_position(parsed):
    # volTPCActive has inline position: x=-0.5, y=0, z=0
    pvs = parsed["volTPC0"]["physvols"]
    pv = next(p for p in pvs if p["vol"] == "volTPCActive")
    np.testing.assert_allclose(pv["pos"], [-0.5, 0.0, 0.0], atol=1e-9)


def test_wireU4_anonymous_inline_position(parsed):
    # Anonymous physvol in volTPCPlaneU0: inline pos → [0, 9.52, 0]
    pvs = parsed["volTPCPlaneU0"]["physvols"]
    pv = next(p for p in pvs if p["vol"] == "volTPCWireU4")
    np.testing.assert_allclose(pv["pos"], [0.0, 9.52, 0.0], atol=1e-9)


def test_worldTPC0_top_inline_position(parsed):
    # volWorld top face: inline position x=15.0
    pvs = parsed["volWorld"]["physvols"]
    pv = next(p for p in pvs if p["name"] == "volTPC0_top")
    np.testing.assert_allclose(pv["pos"], [15.0, 0.0, 0.0], atol=1e-9)


# ── Rotation resolution via rotationref ──────────────────────────────────────

def test_wireU0_rotation(parsed):
    # rWireU: x=30 deg → rx=30°rad, ry=0, rz=0
    pvs = parsed["volTPCPlaneU0"]["physvols"]
    pv = next(p for p in pvs if p["vol"] == "volTPCWireU0")
    np.testing.assert_allclose(pv["rot"], [math.radians(30), 0.0, 0.0], atol=1e-9)


def test_wireV0_rotation(parsed):
    # rWireV: x=-30 deg
    pvs = parsed["volTPCPlaneV0"]["physvols"]
    pv = next(p for p in pvs if p["vol"] == "volTPCWireV0")
    np.testing.assert_allclose(pv["rot"], [math.radians(-30), 0.0, 0.0], atol=1e-9)


def test_identity_rotation(parsed):
    pvs = parsed["volTPC0"]["physvols"]
    pv = next(p for p in pvs if p["vol"] == "volTPCPlaneU0")
    np.testing.assert_allclose(pv["rot"], [0.0, 0.0, 0.0], atol=1e-9)


def test_worldTPC0_bot_rotation(parsed):
    # rPlus180AboutY: y=180 deg
    pvs = parsed["volWorld"]["physvols"]
    pv = next(p for p in pvs if p["name"] == "volTPC0_bot")
    np.testing.assert_allclose(pv["rot"], [0.0, math.radians(180), 0.0], atol=1e-9)


# ── Anonymous physvol naming ──────────────────────────────────────────────────

def test_anonymous_physvol_uses_vol_name(parsed):
    # Wire U4 physvol has no name → name field should be volumeref "volTPCWireU4"
    pvs = parsed["volTPCPlaneU0"]["physvols"]
    pv = next(p for p in pvs if p["vol"] == "volTPCWireU4")
    assert pv["name"] == "volTPCWireU4"


# ── Named physvol preserves given name ───────────────────────────────────────

def test_named_physvol_preserves_name(parsed):
    pvs = parsed["volTPCPlaneU0"]["physvols"]
    pv = next(p for p in pvs if p["vol"] == "volTPCWireU0")
    assert pv["name"] == "volTPCWireU0_pv"


# ── pos and rot are numpy arrays ──────────────────────────────────────────────

def test_pos_is_ndarray(parsed):
    pv = parsed["volTPC0"]["physvols"][0]
    assert isinstance(pv["pos"], np.ndarray)
    assert pv["pos"].shape == (3,)


def test_rot_is_ndarray(parsed):
    pv = parsed["volTPC0"]["physvols"][0]
    assert isinstance(pv["rot"], np.ndarray)
    assert pv["rot"].shape == (3,)
