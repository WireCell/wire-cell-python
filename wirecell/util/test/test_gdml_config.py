"""Tests for detector config JSON schema and load_config() in wirecell.util.gdml."""

import json
import pathlib
import re
import pytest

from wirecell.util.gdml import load_config

_DATA = pathlib.Path(__file__).parent / "data"
_TEST_CFG = _DATA / "test_detector_config.json"


# ── Loading from file ─────────────────────────────────────────────────────────

def test_load_config_from_file_returns_dict():
    cfg = load_config(str(_TEST_CFG))
    assert isinstance(cfg, dict)


def test_load_config_has_role_patterns():
    cfg = load_config(str(_TEST_CFG))
    assert "role_patterns" in cfg
    assert isinstance(cfg["role_patterns"], dict)


def test_load_config_role_patterns_has_required_roles():
    cfg = load_config(str(_TEST_CFG))
    for role in ("wire", "plane", "face", "detector"):
        assert role in cfg["role_patterns"], f"role_patterns missing key: {role}"


def test_load_config_role_patterns_are_valid_regexes():
    cfg = load_config(str(_TEST_CFG))
    for role, pattern in cfg["role_patterns"].items():
        try:
            re.compile(pattern)
        except re.error as exc:
            pytest.fail(f"role_patterns[{role!r}] is not a valid regex: {exc}")


def test_load_config_has_connectivity_mode():
    cfg = load_config(str(_TEST_CFG))
    assert "connectivity_mode" in cfg
    assert cfg["connectivity_mode"] in ("vd", "hd")


def test_load_config_has_nearness_tolerance():
    cfg = load_config(str(_TEST_CFG))
    assert "nearness_tolerance" in cfg
    assert isinstance(cfg["nearness_tolerance"], (int, float))
    assert cfg["nearness_tolerance"] > 0


def test_load_config_role_patterns_match_expected_volumes():
    """Regex patterns in the test config actually match the intended volume names."""
    cfg = load_config(str(_TEST_CFG))
    rp = cfg["role_patterns"]
    assert re.fullmatch(rp["wire"],     "volTPCWireU0")
    assert re.fullmatch(rp["wire"],     "volTPCWireZ12")
    assert re.fullmatch(rp["plane"],    "volTPCPlaneU0")
    assert re.fullmatch(rp["plane"],    "volTPCPlaneZ3")
    assert re.fullmatch(rp["face"],     "volTPC0")
    assert re.fullmatch(rp["detector"], "volCryostat")
    # Non-matching volumes should not match
    assert not re.fullmatch(rp["wire"], "volTPCPlaneU0")
    assert not re.fullmatch(rp["face"], "volCryostat")


def test_load_config_accepts_pathlib_path():
    """load_config should accept a pathlib.Path as well as a str."""
    cfg = load_config(_TEST_CFG)
    assert "role_patterns" in cfg


def test_load_config_returns_independent_copy():
    """Mutating the returned dict must not affect a second call."""
    cfg1 = load_config(str(_TEST_CFG))
    cfg1["connectivity_mode"] = "MUTATED"
    cfg2 = load_config(str(_TEST_CFG))
    assert cfg2["connectivity_mode"] != "MUTATED"


# ── Named (built-in) configs ──────────────────────────────────────────────────

def test_load_config_unknown_name_raises():
    with pytest.raises((ValueError, KeyError)):
        load_config("no_such_detector_xyzzy")


def test_load_config_unknown_name_error_is_informative():
    with pytest.raises((ValueError, KeyError)) as exc_info:
        load_config("no_such_detector_xyzzy")
    msg = str(exc_info.value).lower()
    # The error should mention the unknown name
    assert "no_such_detector_xyzzy" in msg or "unknown" in msg


# ── Validation of required keys ───────────────────────────────────────────────

def test_load_config_missing_role_patterns_raises(tmp_path):
    bad = {"connectivity_mode": "vd", "nearness_tolerance": 0.1}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="role_patterns"):
        load_config(str(p))


def test_load_config_missing_connectivity_mode_raises(tmp_path):
    bad = {
        "role_patterns": {"wire": "volTPCWire.*"},
        "nearness_tolerance": 0.1,
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="connectivity_mode"):
        load_config(str(p))


def test_load_config_missing_nearness_tolerance_raises(tmp_path):
    bad = {
        "role_patterns": {"wire": "volTPCWire.*"},
        "connectivity_mode": "hd",
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="nearness_tolerance"):
        load_config(str(p))


def test_load_config_multiple_missing_keys_raises(tmp_path):
    """An empty JSON object should fail validation."""
    p = tmp_path / "empty.json"
    p.write_text("{}")
    with pytest.raises(ValueError):
        load_config(str(p))
