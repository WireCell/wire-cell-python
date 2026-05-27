"""Tests for parse_solids() in wirecell.util.gdml."""

import xml.etree.ElementTree as ET
import pytest

from wirecell.util.gdml import parse_solids


@pytest.fixture
def gdml_root(minimal_gdml_path):
    return ET.parse(minimal_gdml_path).getroot()


# ── Return structure ──────────────────────────────────────────────────────────

def test_parse_solids_returns_dict(gdml_root):
    result = parse_solids(gdml_root)
    assert isinstance(result, dict)


def test_parse_solids_has_wire_solid_keys(gdml_root):
    result = parse_solids(gdml_root)
    assert "CRMWireU" in result
    assert "CRMWireV" in result
    assert "CRMWireZ" in result


def test_each_entry_has_rmax(gdml_root):
    result = parse_solids(gdml_root)
    for name, dims in result.items():
        assert "rmax" in dims, f"'{name}' missing 'rmax'"


def test_each_entry_has_half_z(gdml_root):
    result = parse_solids(gdml_root)
    for name, dims in result.items():
        assert "half_z" in dims, f"'{name}' missing 'half_z'"


# ── Tube dimensions (mm) from the minimal.gdml solids section ─────────────────

def test_CRMWireU_rmax(gdml_root):
    dims = parse_solids(gdml_root)["CRMWireU"]
    assert abs(dims["rmax"] - 0.076) < 1e-9


def test_CRMWireU_half_z(gdml_root):
    # z=100.0 mm full length → half_z=50.0 mm
    dims = parse_solids(gdml_root)["CRMWireU"]
    assert abs(dims["half_z"] - 50.0) < 1e-9


def test_CRMWireV_rmax(gdml_root):
    dims = parse_solids(gdml_root)["CRMWireV"]
    assert abs(dims["rmax"] - 0.076) < 1e-9


def test_CRMWireV_half_z(gdml_root):
    dims = parse_solids(gdml_root)["CRMWireV"]
    assert abs(dims["half_z"] - 50.0) < 1e-9


def test_CRMWireZ_rmax(gdml_root):
    dims = parse_solids(gdml_root)["CRMWireZ"]
    assert abs(dims["rmax"] - 0.076) < 1e-9


def test_CRMWireZ_half_z(gdml_root):
    # z=50.0 mm full length → half_z=25.0 mm
    dims = parse_solids(gdml_root)["CRMWireZ"]
    assert abs(dims["half_z"] - 25.0) < 1e-9


# ── Values are floats ─────────────────────────────────────────────────────────

def test_rmax_is_float(gdml_root):
    dims = parse_solids(gdml_root)["CRMWireU"]
    assert isinstance(dims["rmax"], float)


def test_half_z_is_float(gdml_root):
    dims = parse_solids(gdml_root)["CRMWireU"]
    assert isinstance(dims["half_z"], float)


# ── Non-tube solids are not included ─────────────────────────────────────────

def test_box_solids_not_in_result(gdml_root):
    result = parse_solids(gdml_root)
    # boxes defined in minimal.gdml: CRMPlaneU, CRMPlaneV, CRMPlaneZ, CRMActive, CRM, World
    for box_name in ("CRMPlaneU", "CRMPlaneV", "CRMPlaneZ", "CRMActive", "CRM", "World"):
        assert box_name not in result, f"box solid '{box_name}' should not be in parse_solids result"
