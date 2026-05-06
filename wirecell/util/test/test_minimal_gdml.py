"""Tests that verify the minimal synthetic GDML fixture is well-formed."""

import xml.etree.ElementTree as ET


def test_synthetic_gdml_exists(minimal_gdml_path):
    assert minimal_gdml_path.exists(), f"Missing fixture: {minimal_gdml_path}"


def test_gdml_parses_as_xml(minimal_gdml_path):
    tree = ET.parse(minimal_gdml_path)
    root = tree.getroot()
    assert root.tag == "gdml"


def test_gdml_has_required_top_level_sections(minimal_gdml_path):
    root = ET.parse(minimal_gdml_path).getroot()
    tags = {child.tag for child in root}
    for required in ("define", "materials", "solids", "structure", "setup"):
        assert required in tags, f"Missing top-level section: {required}"


def test_define_has_positions_and_rotations(minimal_gdml_path):
    root = ET.parse(minimal_gdml_path).getroot()
    define = root.find("define")
    positions = define.findall("position")
    rotations = define.findall("rotation")
    assert len(positions) >= 3, "Need at least 3 named positions in <define>"
    assert len(rotations) >= 3, "Need at least 3 named rotations in <define>"


def test_solids_has_tubes_and_boxes(minimal_gdml_path):
    root = ET.parse(minimal_gdml_path).getroot()
    solids = root.find("solids")
    tubes = solids.findall("tube")
    boxes = solids.findall("box")
    assert len(tubes) >= 3, "Need at least 3 tube solids (one per wire type)"
    assert len(boxes) >= 1, "Need at least 1 box solid (TPC)"


def test_structure_has_tpc_and_plane_volumes(minimal_gdml_path):
    root = ET.parse(minimal_gdml_path).getroot()
    structure = root.find("structure")
    vol_names = {v.get("name") for v in structure.findall("volume")}
    assert any("TPC" in n or "tpc" in n.lower() for n in vol_names), \
        "No TPC volume found"
    assert any("Plane" in n or "plane" in n.lower() for n in vol_names), \
        "No Plane volume found"


def test_structure_has_u_v_and_z_wire_volumes(minimal_gdml_path):
    root = ET.parse(minimal_gdml_path).getroot()
    structure = root.find("structure")
    vol_names = {v.get("name") for v in structure.findall("volume")}
    assert any("WireU" in n or "wireU" in n for n in vol_names), "No U wire volumes"
    assert any("WireV" in n or "wireV" in n for n in vol_names), "No V wire volumes"
    assert any(("WireZ" in n or "wireZ" in n or "WireW" in n) for n in vol_names), \
        "No Z/W wire volumes"


def test_physvols_use_positionref_style(minimal_gdml_path):
    """At least some physvols use <positionref ref=.../>."""
    root = ET.parse(minimal_gdml_path).getroot()
    positionrefs = root.findall(".//physvol/positionref")
    assert len(positionrefs) >= 3, "Need at least 3 physvols using positionref"


def test_physvols_use_inline_position_style(minimal_gdml_path):
    """At least some physvols use inline <position name=... x=... .../> ."""
    root = ET.parse(minimal_gdml_path).getroot()
    inline_positions = root.findall(".//physvol/position")
    assert len(inline_positions) >= 2, "Need at least 2 physvols with inline position"


def test_at_least_one_physvol_has_no_name(minimal_gdml_path):
    """At least one <physvol> element has no 'name' attribute."""
    root = ET.parse(minimal_gdml_path).getroot()
    physvols = root.findall(".//physvol")
    unnamed = [pv for pv in physvols if pv.get("name") is None]
    assert len(unnamed) >= 1, "Need at least one physvol without a name attribute"


def test_each_plane_has_five_wire_physvols(minimal_gdml_path):
    """Each plane logical volume contains exactly 5 wire physvols."""
    root = ET.parse(minimal_gdml_path).getroot()
    structure = root.find("structure")
    for vol in structure.findall("volume"):
        name = vol.get("name", "")
        if "Plane" in name:
            physvols = vol.findall("physvol")
            assert len(physvols) == 5, \
                f"Plane '{name}' has {len(physvols)} physvols, expected 5"


def test_tube_solids_have_z_and_rmax(minimal_gdml_path):
    """Each tube solid has a z (full length) and rmax attribute."""
    root = ET.parse(minimal_gdml_path).getroot()
    solids = root.find("solids")
    for tube in solids.findall("tube"):
        assert tube.get("z") is not None, f"Tube '{tube.get('name')}' missing z"
        assert tube.get("rmax") is not None, f"Tube '{tube.get('name')}' missing rmax"


def test_setup_references_world_volume(minimal_gdml_path):
    root = ET.parse(minimal_gdml_path).getroot()
    setup = root.find("setup")
    world = setup.find("world")
    assert world is not None
    assert world.get("ref") is not None
