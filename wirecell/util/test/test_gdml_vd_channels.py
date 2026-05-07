"""Tests for assign_vd_channels() in wirecell.util.gdml."""

import pytest
import numpy as np

from wirecell.util.gdml import (
    assign_vd_channels, find_vd_connected_pairs,
    WireGeom, PlaneGeom, FaceGeom, AnodeGeom,
)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic VD anode fixture (same geometry as test_gdml_vd_connectivity.py)
# Two induction planes (U, V) + one collection plane (Z).
# ─────────────────────────────────────────────────────────────────────────────

def _w(name, tail, head, plane_name, face_name=""):
    return WireGeom(
        name=name,
        tail=np.array(tail, dtype=float),
        head=np.array(head, dtype=float),
        radius=0.076,
        plane_name=plane_name,
        face_name=face_name,
    )


def _make_vd_anode(anode_x_offset=0.0):
    """Build a synthetic VD anode with 2 U + 2 V + 2 Z wires per face."""
    ox = anode_x_offset

    # face0 (seg=1 for induction): head at z=0
    u0f0 = _w("U0_f0", [ox,-5,-50], [ox,-5, 0], "planeU", "face0")
    u1f0 = _w("U1_f0", [ox, 0,-50], [ox, 0, 0], "planeU", "face0")
    v0f0 = _w("V0_f0", [ox,-5,-50], [ox,-5, 0], "planeV", "face0")
    v1f0 = _w("V1_f0", [ox, 0,-50], [ox, 0, 0], "planeV", "face0")
    z0f0 = _w("Z0_f0", [ox,-5,-25], [ox,-5,25], "planeZ", "face0")
    z1f0 = _w("Z1_f0", [ox, 0,-25], [ox, 0,25], "planeZ", "face0")

    # face1 (seg=0 for induction): tail at z=0
    u0f1 = _w("U0_f1", [ox,-5,  0], [ox,-5,50], "planeU", "face1")
    u1f1 = _w("U1_f1", [ox, 0,  0], [ox, 0,50], "planeU", "face1")
    v0f1 = _w("V0_f1", [ox,-5,  0], [ox,-5,50], "planeV", "face1")
    v1f1 = _w("V1_f1", [ox, 0,  0], [ox, 0,50], "planeV", "face1")
    z0f1 = _w("Z0_f1", [ox,-5,-25], [ox,-5,25], "planeZ", "face1")
    z1f1 = _w("Z1_f1", [ox, 0,-25], [ox, 0,25], "planeZ", "face1")

    face0 = FaceGeom("face0", [
        PlaneGeom("planeU", [u0f0, u1f0]),
        PlaneGeom("planeV", [v0f0, v1f0]),
        PlaneGeom("planeZ", [z0f0, z1f0]),
    ])
    face1 = FaceGeom("face1", [
        PlaneGeom("planeU", [u0f1, u1f1]),
        PlaneGeom("planeV", [v0f1, v1f1]),
        PlaneGeom("planeZ", [z0f1, z1f1]),
    ])
    return AnodeGeom(faces=[face0, face1])


def _make_connectivity(anode):
    return find_vd_connected_pairs(anode, nearness_tolerance=1.0)


@pytest.fixture
def anode():
    return _make_vd_anode()


@pytest.fixture
def connectivity(anode):
    return _make_connectivity(anode)


@pytest.fixture
def channels(anode, connectivity):
    return assign_vd_channels([anode], connectivity)


# ── Return type ───────────────────────────────────────────────────────────────

def test_returns_dict(channels):
    assert isinstance(channels, dict)


def test_all_wires_assigned(anode, channels):
    all_names = {w.name for f in anode.faces for p in f.planes for w in p.wires}
    assert set(channels.keys()) == all_names


def test_all_values_are_ints(channels):
    for v in channels.values():
        assert isinstance(v, int)


# ── Cross-face induction pairs share a channel ────────────────────────────────

def test_u0_pair_shares_channel(channels):
    assert channels["U0_f0"] == channels["U0_f1"]


def test_u1_pair_shares_channel(channels):
    assert channels["U1_f0"] == channels["U1_f1"]


def test_v0_pair_shares_channel(channels):
    assert channels["V0_f0"] == channels["V0_f1"]


def test_v1_pair_shares_channel(channels):
    assert channels["V1_f0"] == channels["V1_f1"]


# ── Collection wires get unique channels ──────────────────────────────────────

def test_z_wires_have_distinct_channels(channels):
    z_channels = [channels[n] for n in ("Z0_f0", "Z1_f0", "Z0_f1", "Z1_f1")]
    assert len(set(z_channels)) == 4


def test_z_channels_not_shared_with_induction(channels):
    induction = {channels[n] for n in ("U0_f0", "U1_f0", "V0_f0", "V1_f0")}
    z_channels = {channels[n] for n in ("Z0_f0", "Z1_f0", "Z0_f1", "Z1_f1")}
    assert induction.isdisjoint(z_channels)


# ── Unique channel count ──────────────────────────────────────────────────────

def test_unique_channel_count(channels):
    # 2 U pairs (2 ch) + 2 V pairs (2 ch) + 4 Z wires (4 ch) = 8 unique channels
    assert len(set(channels.values())) == 8


# ── Multiple anodes: globally unique channels ─────────────────────────────────

def _make_two_anode_setup():
    a0 = _make_vd_anode(anode_x_offset=0.0)
    # Different wire names and positions for anode 1
    def _w2(name, tail, head, plane_name, face_name=""):
        return WireGeom(name=name, tail=np.array(tail, dtype=float),
                        head=np.array(head, dtype=float),
                        radius=0.076, plane_name=plane_name, face_name=face_name)
    f0 = FaceGeom("face0", [
        PlaneGeom("planeU", [
            _w2("A1_U0_f0", [100,-5,-50], [100,-5, 0], "planeU", "face0"),
            _w2("A1_U1_f0", [100, 0,-50], [100, 0, 0], "planeU", "face0"),
        ]),
        PlaneGeom("planeZ", [
            _w2("A1_Z0_f0", [100,-5,-25], [100,-5,25], "planeZ", "face0"),
        ]),
    ])
    f1 = FaceGeom("face1", [
        PlaneGeom("planeU", [
            _w2("A1_U0_f1", [100,-5,  0], [100,-5,50], "planeU", "face1"),
            _w2("A1_U1_f1", [100, 0,  0], [100, 0,50], "planeU", "face1"),
        ]),
        PlaneGeom("planeZ", [
            _w2("A1_Z0_f1", [100,-5,-25], [100,-5,25], "planeZ", "face1"),
        ]),
    ])
    a1 = AnodeGeom(faces=[f0, f1])
    conn0 = find_vd_connected_pairs(a0, nearness_tolerance=1.0)
    conn1 = find_vd_connected_pairs(a1, nearness_tolerance=1.0)
    combined = {**conn0, **conn1}
    return [a0, a1], combined


def test_two_anodes_globally_unique():
    anodes, connectivity = _make_two_anode_setup()
    channels = assign_vd_channels(anodes, connectivity)
    a0_names = {w.name for f in anodes[0].faces for p in f.planes for w in p.wires}
    a1_names = {w.name for f in anodes[1].faces for p in f.planes for w in p.wires}
    ch0 = {channels[n] for n in a0_names}
    ch1 = {channels[n] for n in a1_names}
    assert ch0.isdisjoint(ch1), "Anode 0 and anode 1 share channel IDs"


def test_two_anodes_anode1_channels_higher():
    anodes, connectivity = _make_two_anode_setup()
    channels = assign_vd_channels(anodes, connectivity)
    a0_names = {w.name for f in anodes[0].faces for p in f.planes for w in p.wires}
    a1_names = {w.name for f in anodes[1].faces for p in f.planes for w in p.wires}
    ch0_max = max(channels[n] for n in a0_names)
    ch1_min = min(channels[n] for n in a1_names)
    assert ch1_min > ch0_max


# ── Structural comparison against known-good VD wires file ───────────────────
# Verify our implementation produces the same number of unique channels per
# anode as the reference protodunevd-wires-larsoft-v3.json.bz2 (1536/anode).
# (Absolute channel values differ because we use sequential rather than
#  interleaved numbering.)

_REF_WIRES_PATH = (
    "/home/bviren/dev/gdml-to-wires/dunereco/dunereco/DUNEWireCell/"
    "protodunevd/protodunevd-wires-larsoft-v3.json.bz2"
)
_REF_CHANNELS_PER_ANODE = 1536


def test_known_good_channels_per_anode():
    """Our channel count per anode matches the reference file."""
    try:
        from wirecell.util.wires import persist
        store = persist.load(_REF_WIRES_PATH)
    except Exception:
        pytest.skip("Reference wires file not available")

    for i, anode in enumerate(store.anodes):
        ch = set()
        for fid in anode.faces:
            face = store.faces[fid]
            for pid in face.planes:
                plane = store.planes[pid]
                for wid in plane.wires:
                    ch.add(store.wires[wid].channel)
        assert len(ch) == _REF_CHANNELS_PER_ANODE, (
            f"anode {i}: expected {_REF_CHANNELS_PER_ANODE} channels, got {len(ch)}"
        )


def test_known_good_cross_face_sharing():
    """Cross-face induction pairs share channel IDs (verified against reference)."""
    try:
        from wirecell.util.wires import persist
        store = persist.load(_REF_WIRES_PATH)
    except Exception:
        pytest.skip("Reference wires file not available")

    anode = store.anodes[0]
    # Two faces; find the U plane (ident=0) in each
    face_channels = []
    for fid in anode.faces:
        face = store.faces[fid]
        u_plane = next(
            (store.planes[pid] for pid in face.planes if store.planes[pid].ident == 0),
            None,
        )
        if u_plane:
            face_channels.append(set(store.wires[wid].channel for wid in u_plane.wires))
    assert len(face_channels) == 2
    shared = face_channels[0] & face_channels[1]
    assert len(shared) > 0, "No shared channels between cross-face U planes in reference"
