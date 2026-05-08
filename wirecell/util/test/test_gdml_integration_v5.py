"""Integration test: protodunevd_v5 gdml-to-wires structural smoke test.

No reference wires file exists for v5, so tests check structural counts and
channel assignments rather than comparing coordinates to an external truth.

Marked with pytest.mark.integration so it can be skipped in fast CI runs:

    pytest -m "not integration"   # fast suite (skips this file)
    pytest -m integration         # run only integration tests

The GDML source lives outside this package:

    GDML:  ../dunecore/dunecore/Geometry/gdml/protodunevd_v5_ggd.gdml

Tests are skipped automatically when the file is absent.
"""

import pathlib
import pytest

from wirecell.util.gdml import load_config, convert

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE = pathlib.Path(__file__).parent
_GDML = _HERE.parents[3] / "dunecore/dunecore/Geometry/gdml/protodunevd_v5_ggd.gdml"

pytestmark = pytest.mark.integration


def _skip_if_missing():
    if not _GDML.exists():
        pytest.skip(f"GDML not found: {_GDML}")


@pytest.fixture(scope="module")
def store():
    _skip_if_missing()
    cfg = load_config("protodunevd_v5")
    return convert(_GDML, cfg)


# ── Top-level counts ──────────────────────────────────────────────────────────

def test_anode_count(store):
    assert len(store.anodes) == 8


def test_face_count(store):
    assert len(store.faces) == 16


def test_plane_count(store):
    assert len(store.planes) == 48


def test_total_wire_count(store):
    assert len(store.wires) == 13840


def test_has_one_detector(store):
    assert len(store.detectors) == 1
    assert store.detectors[0].ident == 0


# ── Per-anode structure ───────────────────────────────────────────────────────

def test_each_anode_has_two_faces(store):
    for ai, anode in enumerate(store.anodes):
        assert len(anode.faces) == 2, f"anode {ai} has {len(anode.faces)} faces"


def test_each_face_has_three_planes(store):
    for ai, anode in enumerate(store.anodes):
        for fi in anode.faces:
            face = store.faces[fi]
            assert len(face.planes) == 3, \
                f"anode {ai} face {fi} has {len(face.planes)} planes"


def test_plane_idents_are_0_1_2(store):
    for face in store.faces:
        idents = sorted(store.planes[pi].ident for pi in face.planes)
        assert idents == [0, 1, 2]


def test_anode_idents_sequential(store):
    idents = [a.ident for a in store.anodes]
    assert sorted(idents) == list(range(8))


def test_face_idents_per_anode(store):
    for ai, anode in enumerate(store.anodes):
        fi_idents = sorted(store.faces[fi].ident for fi in anode.faces)
        assert fi_idents == [0, 1], f"anode {ai} face idents: {fi_idents}"


# ── Wire counts per plane ─────────────────────────────────────────────────────
# protodunevd_v5: 292 Z-wires per face, 865 wires total per face (287+286+292).

def test_z_wire_count_per_face(store):
    for ai, anode in enumerate(store.anodes):
        for fi in anode.faces:
            zplane = next(store.planes[pi] for pi in store.faces[fi].planes
                          if store.planes[pi].ident == 2)
            assert len(zplane.wires) == 292, \
                f"anode {ai} face {fi} Z-plane: {len(zplane.wires)} wires"


def test_total_wires_per_face(store):
    for ai, anode in enumerate(store.anodes):
        for fi in anode.faces:
            total = sum(len(store.planes[pi].wires) for pi in store.faces[fi].planes)
            assert total == 865, \
                f"anode {ai} face {fi}: {total} wires (expected 865)"


# ── Channel assignment ────────────────────────────────────────────────────────

def test_channels_unique_per_anode(store):
    for ai, anode in enumerate(store.anodes):
        channels = {store.wires[wi].channel
                    for fi in anode.faces
                    for pi in store.faces[fi].planes
                    for wi in store.planes[pi].wires}
        assert len(channels) == 1536, \
            f"anode {ai}: expected 1536 unique channels, got {len(channels)}"


def test_channels_globally_unique(store):
    anode_channels = []
    for anode in store.anodes:
        ch = {store.wires[wi].channel
              for fi in anode.faces
              for pi in store.faces[fi].planes
              for wi in store.planes[pi].wires}
        anode_channels.append(ch)
    for i in range(len(anode_channels)):
        for j in range(i + 1, len(anode_channels)):
            assert anode_channels[i].isdisjoint(anode_channels[j]), \
                f"anodes {i} and {j} share channel IDs"


def test_channel_range(store):
    all_chans = [w.channel for w in store.wires]
    assert min(all_chans) == 0
    assert max(all_chans) == 12287


def test_channels_are_ints(store):
    for w in store.wires:
        assert isinstance(w.channel, int)


# ── Segment counts ────────────────────────────────────────────────────────────
# VD: U and V planes have cross-face sharing → some wires have segment=1.

def test_segment_values_valid(store):
    for w in store.wires:
        assert w.segment in (0, 1), f"unexpected segment {w.segment}"


def test_cross_face_segments_exist(store):
    seg1_wires = [w for w in store.wires if w.segment == 1]
    assert len(seg1_wires) > 0, "expected some segment-1 wires for cross-face sharing"


# ── All indices in-range ──────────────────────────────────────────────────────

def test_all_indices_in_range(store):
    for det in store.detectors:
        for ai in det.anodes:
            assert 0 <= ai < len(store.anodes)
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


# ── Wire geometry sanity ──────────────────────────────────────────────────────

def test_wire_endpoints_differ(store):
    for w in store.wires:
        t = store.points[w.tail]
        h = store.points[w.head]
        assert (t.x, t.y, t.z) != (h.x, h.y, h.z), \
            f"wire {w.ident} has coincident tail and head"
