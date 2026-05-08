"""Integration test: protodunehd_v8 gdml-to-wires vs known-good wires file.

Marked with pytest.mark.integration so it can be skipped in fast CI runs:

    pytest -m "not integration"   # fast suite (skips this file)
    pytest -m integration         # run only integration tests

Both the GDML source and the reference wires file live outside this package:

    GDML:  ../dunecore/dunecore/Geometry/gdml/protodunehd_v8_refactored.gdml
    REF:   ../dunereco/dunereco/DUNEWireCell/pdhd/protodunehd-wires-larsoft-v1.json.bz2

Tests are skipped automatically when either file is absent.

Channel matching uses 1mm midpoint tolerance because the v8 GDML and the
reference originate from slightly different geometry versions (~0.005 mm
coordinate deltas that straddle the 0.1mm rounding boundary).  99% of wires
match exactly; the remaining ~1% have no reference counterpart within 1mm and
are attributed to genuine coordinate differences between GDML versions.
"""

import pathlib
import json
import bz2

import numpy as np
import pytest

from wirecell.util.gdml import load_config, convert
from wirecell.util.wires import persist

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE = pathlib.Path(__file__).parent
_GDML = _HERE.parents[3] / "dunecore/dunecore/Geometry/gdml/protodunehd_v8_refactored.gdml"
_REF  = _HERE.parents[3] / "dunereco/dunereco/DUNEWireCell/pdhd/protodunehd-wires-larsoft-v1.json.bz2"

pytestmark = pytest.mark.integration


def _skip_if_missing():
    if not _GDML.exists():
        pytest.skip(f"GDML not found: {_GDML}")
    if not _REF.exists():
        pytest.skip(f"Reference wires file not found: {_REF}")


@pytest.fixture(scope="module")
def stores():
    _skip_if_missing()
    cfg = load_config("protodunehd_v8")
    ours = convert(_GDML, cfg)
    with bz2.open(_REF, "rt") as fh:
        ref = persist.fromdict(json.load(fh))
    return ours, ref


# ── Top-level counts ──────────────────────────────────────────────────────────

def test_anode_count(stores):
    ours, ref = stores
    assert len(ours.anodes) == len(ref.anodes) == 4


def test_face_count(stores):
    ours, ref = stores
    assert len(ours.faces) == len(ref.faces) == 8


def test_plane_count(stores):
    ours, ref = stores
    assert len(ours.planes) == len(ref.planes) == 24


def test_total_wire_count(stores):
    ours, ref = stores
    assert len(ours.wires) == len(ref.wires) == 22208


def test_has_one_detector(stores):
    ours, _ = stores
    assert len(ours.detectors) == 1
    assert ours.detectors[0].ident == 0


# ── Per-anode structure ───────────────────────────────────────────────────────

def test_each_anode_has_two_faces(stores):
    ours, _ = stores
    for ai, anode in enumerate(ours.anodes):
        assert len(anode.faces) == 2, f"anode {ai} has {len(anode.faces)} faces"


def test_each_face_has_three_planes(stores):
    ours, _ = stores
    for ai, anode in enumerate(ours.anodes):
        for fi in anode.faces:
            face = ours.faces[fi]
            assert len(face.planes) == 3, \
                f"anode {ai} face {fi} has {len(face.planes)} planes"


def test_plane_idents_are_0_1_2(stores):
    ours, _ = stores
    for face in ours.faces:
        idents = sorted(ours.planes[pi].ident for pi in face.planes)
        assert idents == [0, 1, 2]


def test_anode_idents_sequential(stores):
    ours, _ = stores
    idents = [a.ident for a in ours.anodes]
    assert sorted(idents) == list(range(4))


# ── Channel structure ─────────────────────────────────────────────────────────
# HD: 2560 channels per anode (U:800 + V:800 + W_face0:480 + W_face1:480)

def test_channels_unique_per_anode(stores):
    ours, _ = stores
    for ai, anode in enumerate(ours.anodes):
        channels = {ours.wires[wi].channel
                    for fi in anode.faces
                    for pi in ours.faces[fi].planes
                    for wi in ours.planes[pi].wires}
        assert len(channels) == 2560, \
            f"anode {ai}: expected 2560 unique channels, got {len(channels)}"


def test_channels_globally_unique(stores):
    ours, _ = stores
    anode_channels = []
    for anode in ours.anodes:
        ch = {ours.wires[wi].channel
              for fi in anode.faces
              for pi in ours.faces[fi].planes
              for wi in ours.planes[pi].wires}
        anode_channels.append(ch)
    for i in range(len(anode_channels)):
        for j in range(i + 1, len(anode_channels)):
            assert anode_channels[i].isdisjoint(anode_channels[j]), \
                f"anodes {i} and {j} share channel IDs"


def test_channel_range(stores):
    ours, _ = stores
    all_chans = [w.channel for w in ours.wires]
    assert min(all_chans) == 0
    assert max(all_chans) == 10239


def test_channels_are_ints(stores):
    ours, _ = stores
    for w in ours.wires:
        assert isinstance(w.channel, int)


# ── Segment values ────────────────────────────────────────────────────────────

def test_segment_values_valid(stores):
    ours, _ = stores
    for w in ours.wires:
        assert w.segment in (0, 1, 2), f"unexpected segment {w.segment}"


def test_cross_face_segments_exist(stores):
    """HD U/V planes have cross-face wires (seg=1 and seg=2)."""
    ours, _ = stores
    seg1 = [w for w in ours.wires if w.segment == 1]
    seg2 = [w for w in ours.wires if w.segment == 2]
    assert len(seg1) > 0, "expected seg=1 wires for cross-face induction planes"
    assert len(seg2) > 0, "expected seg=2 wires for cross-face induction planes"


# ── Channel assignment matches reference (midpoint lookup at 1mm tolerance) ───

@pytest.fixture(scope="module")
def ref_lookup(stores):
    """Map wire midpoint (rounded to nearest mm) -> (channel, segment)."""
    _, ref = stores
    lookup = {}
    for w in ref.wires:
        t = ref.points[w.tail]; h = ref.points[w.head]
        key = (round((t.x + h.x) / 2), round((t.y + h.y) / 2), round((t.z + h.z) / 2))
        lookup[key] = (w.channel, w.segment)
    return lookup


def test_channel_assignment_matches_reference(stores, ref_lookup):
    """At least 95% of wires must have matching channel and segment."""
    ours, _ = stores
    ok = wrong = no_match = 0
    for w in ours.wires:
        t = ours.points[w.tail]; h = ours.points[w.head]
        key = (round((t.x + h.x) / 2), round((t.y + h.y) / 2), round((t.z + h.z) / 2))
        if key in ref_lookup:
            rch, rseg = ref_lookup[key]
            if w.channel == rch and w.segment == rseg:
                ok += 1
            else:
                wrong += 1
        else:
            no_match += 1
    total = len(ours.wires)
    match_frac = ok / total
    assert wrong == 0, f"{wrong} wires have wrong channel or segment"
    assert match_frac >= 0.95, \
        f"only {ok}/{total} ({100*match_frac:.1f}%) wires match reference"


# ── All indices in-range ──────────────────────────────────────────────────────

def test_all_indices_in_range(stores):
    ours, _ = stores
    for det in ours.detectors:
        for ai in det.anodes:
            assert 0 <= ai < len(ours.anodes)
    for anode in ours.anodes:
        for fi in anode.faces:
            assert 0 <= fi < len(ours.faces)
            for pi in ours.faces[fi].planes:
                assert 0 <= pi < len(ours.planes)
                for wi in ours.planes[pi].wires:
                    assert 0 <= wi < len(ours.wires)
                    w = ours.wires[wi]
                    assert 0 <= w.tail < len(ours.points)
                    assert 0 <= w.head < len(ours.points)


# ── Wire geometry sanity ──────────────────────────────────────────────────────

def test_wire_endpoints_differ(stores):
    ours, _ = stores
    for w in ours.wires:
        t = ours.points[w.tail]; h = ours.points[w.head]
        assert (t.x, t.y, t.z) != (h.x, h.y, h.z), \
            f"wire {w.ident} has coincident tail and head"
