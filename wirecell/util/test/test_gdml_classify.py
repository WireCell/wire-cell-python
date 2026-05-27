"""Tests for match_role() and classify_volumes() in wirecell.util.gdml."""

import pytest
from wirecell.util.gdml import match_role, classify_volumes

# Patterns matching the test detector config layout
_PATTERNS = {
    "wire":     r"volTPCWire[UVZ]\d+",
    "plane":    r"volTPCPlane[UVZ]\d+",
    "face":     r"volTPC\d+",
    "detector": r"volCryostat",
}

# A representative sample of volume names
_VOL_NAMES = [
    "volTPCWireU0", "volTPCWireU4", "volTPCWireV2", "volTPCWireZ3",
    "volTPCPlaneU0", "volTPCPlaneV0", "volTPCPlaneZ0",
    "volTPC0", "volTPC1",
    "volCryostat",
    "volTPCActive",   # no match
    "volWorld",       # no match
]


# ── match_role ────────────────────────────────────────────────────────────────

def test_match_role_returns_wire_for_wire_volume():
    assert match_role("volTPCWireU0", _PATTERNS) == "wire"


def test_match_role_returns_wire_for_numbered_wire():
    assert match_role("volTPCWireZ12", _PATTERNS) == "wire"


def test_match_role_returns_plane_for_plane_volume():
    assert match_role("volTPCPlaneV0", _PATTERNS) == "plane"


def test_match_role_returns_face_for_tpc_volume():
    assert match_role("volTPC0", _PATTERNS) == "face"


def test_match_role_returns_detector_for_cryostat():
    assert match_role("volCryostat", _PATTERNS) == "detector"


def test_match_role_returns_none_for_unmatched():
    assert match_role("volTPCActive", _PATTERNS) is None


def test_match_role_returns_none_for_world():
    assert match_role("volWorld", _PATTERNS) is None


def test_match_role_returns_none_for_empty_string():
    assert match_role("", _PATTERNS) is None


def test_match_role_no_partial_match():
    # "volTPCPlaneU0" should not match the wire pattern
    assert match_role("volTPCPlaneU0", _PATTERNS) != "wire"


def test_match_role_plane_not_matched_as_face():
    # "volTPCPlaneU0" contains digits but the face pattern is volTPC\d+
    # It should match as "plane" not "face"
    assert match_role("volTPCPlaneU0", _PATTERNS) == "plane"


def test_match_role_accepts_string_patterns():
    # Patterns are plain strings (not pre-compiled); must work without pre-compilation
    patterns = {"wire": r"volTPCWire[UVZ]\d+"}
    assert match_role("volTPCWireU0", patterns) == "wire"


def test_match_role_empty_patterns_returns_none():
    assert match_role("volTPCWireU0", {}) is None


# ── classify_volumes ──────────────────────────────────────────────────────────

def test_classify_volumes_returns_dict():
    result = classify_volumes(_VOL_NAMES, _PATTERNS)
    assert isinstance(result, dict)


def test_classify_volumes_wire_role_present():
    result = classify_volumes(_VOL_NAMES, _PATTERNS)
    assert "wire" in result


def test_classify_volumes_wire_names():
    result = classify_volumes(_VOL_NAMES, _PATTERNS)
    wire_names = set(result["wire"])
    assert "volTPCWireU0" in wire_names
    assert "volTPCWireU4" in wire_names
    assert "volTPCWireV2" in wire_names
    assert "volTPCWireZ3" in wire_names


def test_classify_volumes_plane_names():
    result = classify_volumes(_VOL_NAMES, _PATTERNS)
    plane_names = set(result["plane"])
    assert "volTPCPlaneU0" in plane_names
    assert "volTPCPlaneV0" in plane_names
    assert "volTPCPlaneZ0" in plane_names


def test_classify_volumes_face_names():
    result = classify_volumes(_VOL_NAMES, _PATTERNS)
    face_names = set(result["face"])
    assert "volTPC0" in face_names
    assert "volTPC1" in face_names


def test_classify_volumes_detector_names():
    result = classify_volumes(_VOL_NAMES, _PATTERNS)
    assert "volCryostat" in result["detector"]


def test_classify_volumes_unmatched_not_in_result():
    result = classify_volumes(_VOL_NAMES, _PATTERNS)
    for names in result.values():
        assert "volTPCActive" not in names
        assert "volWorld" not in names


def test_classify_volumes_no_none_key():
    result = classify_volumes(_VOL_NAMES, _PATTERNS)
    assert None not in result


def test_classify_volumes_count():
    result = classify_volumes(_VOL_NAMES, _PATTERNS)
    total = sum(len(v) for v in result.values())
    # 4 wires + 3 planes + 2 faces + 1 detector = 10 matched out of 12
    assert total == 10


def test_classify_volumes_empty_input():
    result = classify_volumes([], _PATTERNS)
    assert result == {}


def test_classify_volumes_all_unmatched():
    result = classify_volumes(["volFoo", "volBar"], _PATTERNS)
    assert result == {}
