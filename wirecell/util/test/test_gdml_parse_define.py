"""Tests for parse_define() in wirecell.util.gdml."""

import math
import xml.etree.ElementTree as ET
import pytest
import numpy as np

from wirecell.util.gdml import parse_define


@pytest.fixture
def gdml_root(minimal_gdml_path):
    return ET.parse(minimal_gdml_path).getroot()


# ── Return structure ──────────────────────────────────────────────────────────

def test_parse_define_returns_dict(gdml_root):
    result = parse_define(gdml_root)
    assert isinstance(result, dict)


def test_parse_define_has_positions_key(gdml_root):
    result = parse_define(gdml_root)
    assert "positions" in result


def test_parse_define_has_rotations_key(gdml_root):
    result = parse_define(gdml_root)
    assert "rotations" in result


def test_positions_is_dict(gdml_root):
    result = parse_define(gdml_root)
    assert isinstance(result["positions"], dict)


def test_rotations_is_dict(gdml_root):
    result = parse_define(gdml_root)
    assert isinstance(result["rotations"], dict)


# ── Named positions present ───────────────────────────────────────────────────

def test_positions_contains_posPlaneU0(gdml_root):
    positions = parse_define(gdml_root)["positions"]
    assert "posPlaneU0" in positions


def test_positions_contains_posWireU0(gdml_root):
    positions = parse_define(gdml_root)["positions"]
    assert "posWireU0" in positions


def test_positions_contains_posWireZ4(gdml_root):
    positions = parse_define(gdml_root)["positions"]
    assert "posWireZ4" in positions


# ── Position values (mm, from the minimal.gdml define section) ────────────────

def test_posPlaneU0_value(gdml_root):
    pos = parse_define(gdml_root)["positions"]["posPlaneU0"]
    np.testing.assert_allclose(pos, [-4.76, 0.0, 0.0], atol=1e-9)


def test_posPlaneV0_value(gdml_root):
    pos = parse_define(gdml_root)["positions"]["posPlaneV0"]
    np.testing.assert_allclose(pos, [0.0, 0.0, 0.0], atol=1e-9)


def test_posPlaneZ0_value(gdml_root):
    pos = parse_define(gdml_root)["positions"]["posPlaneZ0"]
    np.testing.assert_allclose(pos, [4.76, 0.0, 0.0], atol=1e-9)


def test_posWireU0_value(gdml_root):
    pos = parse_define(gdml_root)["positions"]["posWireU0"]
    np.testing.assert_allclose(pos, [0.0, -9.52, 0.0], atol=1e-9)


def test_posWireU3_value(gdml_root):
    # posWireU4 is inline (not in <define>); posWireU3 is named
    pos = parse_define(gdml_root)["positions"]["posWireU3"]
    np.testing.assert_allclose(pos, [0.0, 4.76, 0.0], atol=1e-9)


def test_posWireZ0_value(gdml_root):
    pos = parse_define(gdml_root)["positions"]["posWireZ0"]
    np.testing.assert_allclose(pos, [0.0, 0.0, -9.52], atol=1e-9)


def test_posWireZ4_value(gdml_root):
    pos = parse_define(gdml_root)["positions"]["posWireZ4"]
    np.testing.assert_allclose(pos, [0.0, 0.0, 9.52], atol=1e-9)


# ── Position values are numpy arrays ─────────────────────────────────────────

def test_position_is_ndarray(gdml_root):
    pos = parse_define(gdml_root)["positions"]["posPlaneU0"]
    assert isinstance(pos, np.ndarray)
    assert pos.shape == (3,)


# ── Named rotations present ───────────────────────────────────────────────────

def test_rotations_contains_rIdentity(gdml_root):
    rotations = parse_define(gdml_root)["rotations"]
    assert "rIdentity" in rotations


def test_rotations_contains_rWireU(gdml_root):
    rotations = parse_define(gdml_root)["rotations"]
    assert "rWireU" in rotations


def test_rotations_contains_rWireV(gdml_root):
    rotations = parse_define(gdml_root)["rotations"]
    assert "rWireV" in rotations


def test_rotations_contains_rWireW(gdml_root):
    rotations = parse_define(gdml_root)["rotations"]
    assert "rWireW" in rotations


# ── Rotation values (stored in degrees in GDML; returned in radians) ──────────

def test_rIdentity_value(gdml_root):
    rot = parse_define(gdml_root)["rotations"]["rIdentity"]
    np.testing.assert_allclose(rot, [0.0, 0.0, 0.0], atol=1e-9)


def test_rWireU_value(gdml_root):
    # rWireU: x=30 deg, y=0, z=0
    rot = parse_define(gdml_root)["rotations"]["rWireU"]
    np.testing.assert_allclose(rot, [math.radians(30), 0.0, 0.0], atol=1e-9)


def test_rWireV_value(gdml_root):
    # rWireV: x=-30 deg, y=0, z=0
    rot = parse_define(gdml_root)["rotations"]["rWireV"]
    np.testing.assert_allclose(rot, [math.radians(-30), 0.0, 0.0], atol=1e-9)


def test_rWireW_value(gdml_root):
    # rWireW: x=90 deg, y=0, z=0
    rot = parse_define(gdml_root)["rotations"]["rWireW"]
    np.testing.assert_allclose(rot, [math.radians(90), 0.0, 0.0], atol=1e-9)


def test_rPlus180AboutY_value(gdml_root):
    # rPlus180AboutY: x=0, y=180 deg, z=0
    rot = parse_define(gdml_root)["rotations"]["rPlus180AboutY"]
    np.testing.assert_allclose(rot, [0.0, math.radians(180), 0.0], atol=1e-9)


# ── Rotation values are numpy arrays ─────────────────────────────────────────

def test_rotation_is_ndarray(gdml_root):
    rot = parse_define(gdml_root)["rotations"]["rWireU"]
    assert isinstance(rot, np.ndarray)
    assert rot.shape == (3,)
