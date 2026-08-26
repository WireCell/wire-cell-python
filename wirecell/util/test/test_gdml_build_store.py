"""Tests for build_store() in wirecell.util.gdml."""

import pytest
import numpy as np

from wirecell.util.gdml import (
    build_store, assign_vd_channels, find_vd_connected_pairs,
    WireGeom, PlaneGeom, FaceGeom, AnodeGeom,
)
from wirecell.util.wires import schema as wschema, persist


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic VD anode (same geometry as other gdml tests)
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


def _make_anode(prefix="", ox=0.0):
    p = prefix
    face0 = FaceGeom(f"{p}face0", [
        PlaneGeom(f"{p}planeU", [
            _w(f"{p}U0_f0", [ox,-5,-50], [ox,-5, 0], f"{p}planeU", f"{p}face0"),
            _w(f"{p}U1_f0", [ox, 0,-50], [ox, 0, 0], f"{p}planeU", f"{p}face0"),
        ]),
        PlaneGeom(f"{p}planeV", [
            _w(f"{p}V0_f0", [ox,-5,-50], [ox,-5, 0], f"{p}planeV", f"{p}face0"),
            _w(f"{p}V1_f0", [ox, 0,-50], [ox, 0, 0], f"{p}planeV", f"{p}face0"),
        ]),
        PlaneGeom(f"{p}planeZ", [
            _w(f"{p}Z0_f0", [ox,-5,-25], [ox,-5,25], f"{p}planeZ", f"{p}face0"),
            _w(f"{p}Z1_f0", [ox, 0,-25], [ox, 0,25], f"{p}planeZ", f"{p}face0"),
        ]),
    ])
    face1 = FaceGeom(f"{p}face1", [
        PlaneGeom(f"{p}planeU", [
            _w(f"{p}U0_f1", [ox,-5,  0], [ox,-5,50], f"{p}planeU", f"{p}face1"),
            _w(f"{p}U1_f1", [ox, 0,  0], [ox, 0,50], f"{p}planeU", f"{p}face1"),
        ]),
        PlaneGeom(f"{p}planeV", [
            _w(f"{p}V0_f1", [ox,-5,  0], [ox,-5,50], f"{p}planeV", f"{p}face1"),
            _w(f"{p}V1_f1", [ox, 0,  0], [ox, 0,50], f"{p}planeV", f"{p}face1"),
        ]),
        PlaneGeom(f"{p}planeZ", [
            _w(f"{p}Z0_f1", [ox,-5,-25], [ox,-5,25], f"{p}planeZ", f"{p}face1"),
            _w(f"{p}Z1_f1", [ox, 0,-25], [ox, 0,25], f"{p}planeZ", f"{p}face1"),
        ]),
    ])
    return AnodeGeom(faces=[face0, face1])


def _build(anodes):
    conn = {}
    for a in anodes:
        conn.update(find_vd_connected_pairs(a, nearness_tolerance=1.0))
    channels = assign_vd_channels(anodes, conn)
    channel_map = {n: (channels[n], conn.get(n, {}).get("segment", 0)) for n in channels}
    return build_store(anodes, channel_map)


@pytest.fixture
def single_anode():
    return _make_anode()


@pytest.fixture
def store(single_anode):
    return _build([single_anode])


# ── Return type ───────────────────────────────────────────────────────────────

def test_returns_store(store):
    assert isinstance(store, wschema.Store)


def test_store_has_one_detector(store):
    assert len(store.detectors) == 1


def test_store_has_one_anode(store):
    assert len(store.anodes) == 1


def test_store_has_two_faces(store):
    assert len(store.faces) == 2


def test_store_has_six_planes(store):
    # 2 faces × 3 planes = 6
    assert len(store.planes) == 6


def test_store_has_twelve_wires(store):
    # 2 faces × 3 planes × 2 wires = 12
    assert len(store.wires) == 12


def test_store_has_points(store):
    assert len(store.points) > 0


# ── Hierarchy indices are in-range ────────────────────────────────────────────

def test_detector_anode_indices_in_range(store):
    for ai in store.detectors[0].anodes:
        assert 0 <= ai < len(store.anodes)


def test_anode_face_indices_in_range(store):
    for anode in store.anodes:
        for fi in anode.faces:
            assert 0 <= fi < len(store.faces)


def test_face_plane_indices_in_range(store):
    for face in store.faces:
        for pi in face.planes:
            assert 0 <= pi < len(store.planes)


def test_plane_wire_indices_in_range(store):
    for plane in store.planes:
        for wi in plane.wires:
            assert 0 <= wi < len(store.wires)


def test_wire_point_indices_in_range(store):
    for wire in store.wires:
        assert 0 <= wire.tail < len(store.points)
        assert 0 <= wire.head < len(store.points)


# ── Ident values ──────────────────────────────────────────────────────────────

def test_anode_ident(store):
    assert store.anodes[0].ident == 0


def test_face_idents(store):
    anode = store.anodes[0]
    idents = sorted(store.faces[fi].ident for fi in anode.faces)
    assert idents == [0, 1]


def test_plane_idents_per_face(store):
    for face in store.faces:
        idents = sorted(store.planes[pi].ident for pi in face.planes)
        assert idents == [0, 1, 2], f"Plane idents should be 0,1,2; got {idents}"


def test_wire_idents_per_plane(store):
    for plane in store.planes:
        idents = sorted(store.wires[wi].ident for wi in plane.wires)
        assert idents == list(range(len(plane.wires)))


# ── Channel and segment correctness ──────────────────────────────────────────

def test_cross_face_pairs_share_channel(store):
    # In each face there are U and V induction planes (ident 0, 1).
    # Cross-face pairs must share the same channel.
    anode = store.anodes[0]
    face_channels = {}
    for fi in anode.faces:
        face = store.faces[fi]
        face_ident = face.ident
        for pi in face.planes:
            plane = store.planes[pi]
            if plane.ident in (0, 1):  # U and V
                for wi in plane.wires:
                    w = store.wires[wi]
                    face_channels.setdefault((face_ident, plane.ident, w.ident), w.channel)

    # Each (plane_ident, wire_ident) pair must map to same channel across both faces
    # face 0 is seg=1 partner of face 1 (seg=0)
    seen = {}
    for (fi, pi, wi), ch in face_channels.items():
        key = (pi, wi)
        if key in seen:
            assert seen[key] == ch, f"plane {pi} wire {wi} face {fi}: channel mismatch"
        else:
            seen[key] = ch


def test_collection_wires_unique_channels(store):
    # Z planes (ident=2) must all have distinct channels
    z_channels = []
    for plane in store.planes:
        if plane.ident == 2:
            for wi in plane.wires:
                z_channels.append(store.wires[wi].channel)
    assert len(set(z_channels)) == len(z_channels)


def test_all_channels_are_ints(store):
    for wire in store.wires:
        assert isinstance(wire.channel, int)


# ── Wires per plane in pitch order ───────────────────────────────────────────

def test_z_plane_wires_in_pitch_order(store):
    for plane in store.planes:
        if plane.ident != 2:
            continue
        mids = []
        for wi in plane.wires:
            w = store.wires[wi]
            t = store.points[w.tail]
            h = store.points[w.head]
            mids.append(0.5 * (t.z + h.z))
        assert mids == sorted(mids), f"Z wires not in ascending Z order: {mids}"


# ── Two-anode store ───────────────────────────────────────────────────────────

def test_two_anode_store_counts():
    a0 = _make_anode(prefix="A0_", ox=0.0)
    a1 = _make_anode(prefix="A1_", ox=100.0)
    store = _build([a0, a1])
    assert len(store.anodes) == 2
    assert len(store.faces) == 4
    assert len(store.planes) == 12
    assert len(store.wires) == 24


def test_two_anode_channel_disjoint():
    a0 = _make_anode(prefix="A0_", ox=0.0)
    a1 = _make_anode(prefix="A1_", ox=100.0)
    store = _build([a0, a1])
    ch0 = {store.wires[wi].channel for pi in store.faces[store.anodes[0].faces[0]].planes
           for wi in store.planes[pi].wires}
    ch1 = {store.wires[wi].channel for pi in store.faces[store.anodes[1].faces[0]].planes
           for wi in store.planes[pi].wires}
    # ch0 and ch1 may share channels for cross-face induction; restrict to
    # channels of wires in the SAME face (ch0 from anode0.face0 only)
    # Check that anode-level channel sets are disjoint
    all_ch0 = {store.wires[wi].channel
               for fi in store.anodes[0].faces
               for pi in store.faces[fi].planes
               for wi in store.planes[pi].wires}
    all_ch1 = {store.wires[wi].channel
               for fi in store.anodes[1].faces
               for pi in store.faces[fi].planes
               for wi in store.planes[pi].wires}
    assert all_ch0.isdisjoint(all_ch1)


# ── Point deduplication ───────────────────────────────────────────────────────

def test_point_deduplication():
    # Cross-face wires share an endpoint; deduplicated store has fewer points
    # than 2 * num_wires.
    anode = _make_anode()
    store = _build([anode])
    num_wires = len(store.wires)
    # With perfect deduplication, shared endpoints are stored once.
    assert len(store.points) < 2 * num_wires


# ── Round-trip through persist.todict / fromdict ──────────────────────────────

def test_roundtrip_todict_fromdict(store):
    d = persist.todict(store)
    store2 = persist.fromdict(d)
    assert isinstance(store2, wschema.Store)
    assert len(store2.anodes) == len(store.anodes)
    assert len(store2.wires) == len(store.wires)
    assert len(store2.points) == len(store.points)


def test_roundtrip_preserves_channels(store):
    d = persist.todict(store)
    store2 = persist.fromdict(d)
    orig_channels = sorted(w.channel for w in store.wires)
    rt_channels   = sorted(w.channel for w in store2.wires)
    assert orig_channels == rt_channels


def test_roundtrip_preserves_points(store):
    d = persist.todict(store)
    store2 = persist.fromdict(d)
    orig_pts = sorted((p.x, p.y, p.z) for p in store.points)
    rt_pts   = sorted((p.x, p.y, p.z) for p in store2.points)
    assert orig_pts == rt_pts
