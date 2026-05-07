"""Smoke tests for convert() and the gdml-to-wires CLI command."""

import json
import pathlib
import pytest
from click.testing import CliRunner

from wirecell.util.gdml import convert, load_config
from wirecell.util.wires import schema as wschema, persist

_DATA = pathlib.Path(__file__).parent / "data"
_MINIMAL_GDML = _DATA / "minimal.gdml"
_TEST_CONFIG = _DATA / "test_detector_config.json"


# ── convert() ────────────────────────────────────────────────────────────────

@pytest.fixture
def store(minimal_gdml_path):
    cfg = load_config(str(_TEST_CONFIG))
    return convert(minimal_gdml_path, cfg)


def test_convert_returns_store(store):
    assert isinstance(store, wschema.Store)


def test_convert_has_one_detector(store):
    assert len(store.detectors) == 1


def test_convert_has_one_anode(store):
    # minimal.gdml: 1 TPC LV placed twice → 1 anode
    assert len(store.anodes) == 1


def test_convert_has_two_faces(store):
    assert len(store.faces) == 2


def test_convert_has_planes(store):
    assert len(store.planes) > 0


def test_convert_has_wires(store):
    assert len(store.wires) > 0


def test_convert_has_points(store):
    assert len(store.points) > 0


def test_convert_three_planes_per_face(store):
    # minimal.gdml has U, V, Z planes → 3 per face
    anode = store.anodes[0]
    for fi in anode.faces:
        face = store.faces[fi]
        assert len(face.planes) == 3


def test_convert_five_wires_per_plane(store):
    # minimal.gdml has 5 wires per plane
    for plane in store.planes:
        assert len(plane.wires) == 5


def test_convert_all_indices_in_range(store):
    for anode in store.anodes:
        for fi in anode.faces:
            assert 0 <= fi < len(store.faces)
            for pi in store.faces[fi].planes:
                assert 0 <= pi < len(store.planes)
                for wi in store.planes[pi].wires:
                    assert 0 <= wi < len(store.wires)
                    w = store.wires[wi]
                    assert 0 <= w.tail < len(store.points)
                    assert 0 <= w.head < len(store.points)


def test_convert_plane_idents_uvw(store):
    for face in store.faces:
        idents = sorted(store.planes[pi].ident for pi in face.planes)
        assert idents == [0, 1, 2]


def test_convert_channels_are_ints(store):
    for wire in store.wires:
        assert isinstance(wire.channel, int)


def test_convert_unknown_mode_raises():
    cfg = load_config(str(_TEST_CONFIG))
    bad_cfg = dict(cfg, connectivity_mode="hd")
    with pytest.raises(NotImplementedError):
        convert(_MINIMAL_GDML, bad_cfg)


def test_convert_default_root_vol(minimal_gdml_path):
    """convert() auto-detects root vol from <setup> without explicit root_vol."""
    cfg = load_config(str(_TEST_CONFIG))
    store = convert(minimal_gdml_path, cfg)
    assert len(store.wires) > 0


# ── gdml-to-wires CLI ─────────────────────────────────────────────────────────

def test_cli_runs_successfully(tmp_path, minimal_gdml_path):
    from wirecell.util.__main__ import cli
    runner = CliRunner()
    out_file = str(tmp_path / "out.json")
    result = runner.invoke(cli, [
        "gdml-to-wires",
        "-d", str(_TEST_CONFIG),
        "-o", out_file,
        str(minimal_gdml_path),
    ])
    assert result.exit_code == 0, result.output


def test_cli_output_is_valid_json(tmp_path, minimal_gdml_path):
    from wirecell.util.__main__ import cli
    runner = CliRunner()
    out_file = tmp_path / "out.json"
    runner.invoke(cli, [
        "gdml-to-wires",
        "-d", str(_TEST_CONFIG),
        "-o", str(out_file),
        str(minimal_gdml_path),
    ])
    data = json.loads(out_file.read_text())
    assert isinstance(data, dict)


def test_cli_output_has_store_key(tmp_path, minimal_gdml_path):
    from wirecell.util.__main__ import cli
    runner = CliRunner()
    out_file = tmp_path / "out.json"
    runner.invoke(cli, [
        "gdml-to-wires",
        "-d", str(_TEST_CONFIG),
        "-o", str(out_file),
        str(minimal_gdml_path),
    ])
    data = json.loads(out_file.read_text())
    assert "Store" in data


def test_cli_output_roundtrips(tmp_path, minimal_gdml_path):
    from wirecell.util.__main__ import cli
    runner = CliRunner()
    out_file = tmp_path / "out.json"
    runner.invoke(cli, [
        "gdml-to-wires",
        "-d", str(_TEST_CONFIG),
        "-o", str(out_file),
        str(minimal_gdml_path),
    ])
    store2 = persist.load(str(out_file))
    assert isinstance(store2, wschema.Store)
    assert len(store2.anodes) == 1
    assert len(store2.wires) > 0


def test_cli_default_output_name(tmp_path, minimal_gdml_path):
    """Without -o, output is placed beside the GDML with .json extension."""
    import shutil
    from wirecell.util.__main__ import cli
    gdml_copy = tmp_path / "minimal.gdml"
    shutil.copy(minimal_gdml_path, gdml_copy)
    runner = CliRunner()
    result = runner.invoke(cli, [
        "gdml-to-wires",
        "-d", str(_TEST_CONFIG),
        str(gdml_copy),
    ])
    assert result.exit_code == 0, result.output
    expected = tmp_path / "minimal.json"
    assert expected.exists()
