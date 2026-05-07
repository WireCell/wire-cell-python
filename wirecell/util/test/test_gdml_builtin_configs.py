"""Tests for BUILTIN_CONFIGS and load_config() in wirecell.util.gdml."""

import pytest

from wirecell.util.gdml import (
    BUILTIN_CONFIGS, load_config, classify_volumes,
)


# ── BUILTIN_CONFIGS presence ──────────────────────────────────────────────────

def test_protodunevd_v4_present():
    assert "protodunevd_v4" in BUILTIN_CONFIGS


def test_protodunevd_v5_present():
    assert "protodunevd_v5" in BUILTIN_CONFIGS


def test_v4_has_required_keys():
    cfg = BUILTIN_CONFIGS["protodunevd_v4"]
    assert "role_patterns" in cfg
    assert "connectivity_mode" in cfg
    assert "nearness_tolerance" in cfg


def test_v5_has_required_keys():
    cfg = BUILTIN_CONFIGS["protodunevd_v5"]
    assert "role_patterns" in cfg
    assert "connectivity_mode" in cfg
    assert "nearness_tolerance" in cfg


def test_v4_connectivity_mode_is_vd():
    assert BUILTIN_CONFIGS["protodunevd_v4"]["connectivity_mode"] == "vd"


def test_v5_connectivity_mode_is_vd():
    assert BUILTIN_CONFIGS["protodunevd_v5"]["connectivity_mode"] == "vd"


def test_v4_role_patterns_has_four_roles():
    roles = set(BUILTIN_CONFIGS["protodunevd_v4"]["role_patterns"].keys())
    assert roles == {"wire", "plane", "face", "detector"}


def test_v5_role_patterns_has_four_roles():
    roles = set(BUILTIN_CONFIGS["protodunevd_v5"]["role_patterns"].keys())
    assert roles == {"wire", "plane", "face", "detector"}


# ── load_config by name ───────────────────────────────────────────────────────

def test_load_config_v4_by_name():
    cfg = load_config("protodunevd_v4")
    assert cfg["connectivity_mode"] == "vd"


def test_load_config_v5_by_name():
    cfg = load_config("protodunevd_v5")
    assert cfg["connectivity_mode"] == "vd"


def test_load_config_returns_copy():
    cfg1 = load_config("protodunevd_v4")
    cfg1["injected"] = True
    cfg2 = load_config("protodunevd_v4")
    assert "injected" not in cfg2


def test_load_config_unknown_raises():
    with pytest.raises((ValueError, KeyError)):
        load_config("nonexistent_detector")


# ── classify_volumes: protodunevd_v4 volume names ────────────────────────────

_V4_NAMES = [
    "volTPCWireU0",
    "volTPCWireV999",
    "volTPCWireZ3",
    "volTPCPlaneU0",
    "volTPCPlaneV2",
    "volTPCPlaneZ3",
    "volTPC0",
    "volTPC3",
    "volCryostat",
]


@pytest.fixture
def v4_cfg():
    return load_config("protodunevd_v4")


def test_v4_wire_names_classified(v4_cfg):
    roles = classify_volumes(_V4_NAMES, v4_cfg["role_patterns"])
    assert "volTPCWireU0" in roles["wire"]
    assert "volTPCWireV999" in roles["wire"]
    assert "volTPCWireZ3" in roles["wire"]


def test_v4_plane_names_classified(v4_cfg):
    roles = classify_volumes(_V4_NAMES, v4_cfg["role_patterns"])
    assert "volTPCPlaneU0" in roles["plane"]
    assert "volTPCPlaneV2" in roles["plane"]
    assert "volTPCPlaneZ3" in roles["plane"]


def test_v4_face_names_classified(v4_cfg):
    roles = classify_volumes(_V4_NAMES, v4_cfg["role_patterns"])
    assert "volTPC0" in roles["face"]
    assert "volTPC3" in roles["face"]


def test_v4_detector_name_classified(v4_cfg):
    roles = classify_volumes(_V4_NAMES, v4_cfg["role_patterns"])
    assert "volCryostat" in roles["detector"]


def test_v4_no_false_positives(v4_cfg):
    # v5-style names must NOT match v4 patterns
    v5_names = ["volTPCWireU0_0", "volTPCPlaneU_0", "volTPC_0"]
    roles = classify_volumes(v5_names, v4_cfg["role_patterns"])
    all_matched = {n for names in roles.values() for n in names}
    for name in v5_names:
        assert name not in all_matched, f"{name!r} should not match v4 patterns"


# ── classify_volumes: protodunevd_v5 volume names ────────────────────────────

_V5_NAMES = [
    "volTPCWireU0_0",
    "volTPCWireV100_3",
    "volTPCWireZ_0",
    "volTPCWireZ_3",
    "volTPCPlaneU_0",
    "volTPCPlaneV_2",
    "volTPCPlaneZ_3",
    "volTPC_0",
    "volTPC_3",
    "volCryostat",
]


@pytest.fixture
def v5_cfg():
    return load_config("protodunevd_v5")


def test_v5_wire_names_classified(v5_cfg):
    roles = classify_volumes(_V5_NAMES, v5_cfg["role_patterns"])
    assert "volTPCWireU0_0" in roles["wire"]
    assert "volTPCWireV100_3" in roles["wire"]
    assert "volTPCWireZ_0" in roles["wire"]
    assert "volTPCWireZ_3" in roles["wire"]


def test_v5_plane_names_classified(v5_cfg):
    roles = classify_volumes(_V5_NAMES, v5_cfg["role_patterns"])
    assert "volTPCPlaneU_0" in roles["plane"]
    assert "volTPCPlaneV_2" in roles["plane"]
    assert "volTPCPlaneZ_3" in roles["plane"]


def test_v5_face_names_classified(v5_cfg):
    roles = classify_volumes(_V5_NAMES, v5_cfg["role_patterns"])
    assert "volTPC_0" in roles["face"]
    assert "volTPC_3" in roles["face"]


def test_v5_detector_name_classified(v5_cfg):
    roles = classify_volumes(_V5_NAMES, v5_cfg["role_patterns"])
    assert "volCryostat" in roles["detector"]


def test_v5_no_false_positives(v5_cfg):
    # v4-style names must NOT match v5 patterns
    v4_names = ["volTPCWireU0", "volTPCPlaneU0", "volTPC0"]
    roles = classify_volumes(v4_names, v5_cfg["role_patterns"])
    all_matched = {n for names in roles.values() for n in names}
    for name in v4_names:
        assert name not in all_matched, f"{name!r} should not match v5 patterns"
