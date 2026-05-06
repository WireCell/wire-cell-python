"""Tests for VD channel and segment assignment in wirecell.util.gdml."""

import numpy as np
import pytest
from wirecell.util.gdml import (
    WireGeom, PlaneGeom, FaceGeom,
    assign_vd_channels,
)


def make_wire(name, tail, head, plane_name="volTPCPlaneU0"):
    return WireGeom(name=name,
                    tail=np.array(tail, float),
                    head=np.array(head, float),
                    radius=0.15,
                    plane_name=plane_name)


# ── Shared fixtures ────────────────────────────────────────────────────────────

def build_two_face_scenario():
    """
    face0 U-plane: f0_u_w0 (matched) + f0_u_w1 (standalone)
    face1 U-plane: f1_u_w0 (matched)  + f1_u_w1 (standalone)
    face0 Z-plane: f0_z_w0 (collection, standalone)
    face1 Z-plane: f1_z_w0 (collection, standalone)

    Matching: f0_u_w0.head == f1_u_w0.tail at (0, 0, 100).
    """
    junction = (0.0, 0.0, 100.0)
    face0 = FaceGeom(name="face0", planes=[
        PlaneGeom(name="volTPCPlaneU0", wires=[
            make_wire("f0_u_w0", tail=(0, -100, 0), head=junction),  # matched
            make_wire("f0_u_w1", tail=(0, -90, 0),  head=(0, 0, 80)),  # standalone
        ]),
        PlaneGeom(name="volTPCPlaneZ0", wires=[
            make_wire("f0_z_w0", tail=(0, -100, 50), head=(0, 0, 50),
                      plane_name="volTPCPlaneZ0"),
        ]),
    ])
    face1 = FaceGeom(name="face1", planes=[
        PlaneGeom(name="volTPCPlaneU0", wires=[
            make_wire("f1_u_w0", tail=junction,     head=(0, 100, 200)),  # matched
            make_wire("f1_u_w1", tail=(0, 50, 999), head=(0, 100, 999)),  # standalone
        ]),
        PlaneGeom(name="volTPCPlaneZ0", wires=[
            make_wire("f1_z_w0", tail=(0, 0, 50), head=(0, 100, 50),
                      plane_name="volTPCPlaneZ0"),
        ]),
    ])
    return face0, face1


def all_wires(face0, face1):
    ws = {}
    for face in (face0, face1):
        for plane in face.planes:
            for w in plane.wires:
                ws[w.name] = w
    return ws


# ── WireGeom field tests ───────────────────────────────────────────────────────

def test_wiregeom_channel_defaults_none():
    w = WireGeom(name="w", tail=None, head=None, radius=0.15, plane_name="p")
    assert w.channel is None
    assert w.segment is None


def test_wiregeom_channel_can_be_set():
    w = WireGeom(name="w", tail=None, head=None, radius=0.15, plane_name="p",
                 channel=42, segment=1)
    assert w.channel == 42
    assert w.segment == 1


# ── Channel assignment tests ───────────────────────────────────────────────────

def test_cross_face_pair_shares_channel():
    face0, face1 = build_two_face_scenario()
    assign_vd_channels(face0, face1)
    ws = all_wires(face0, face1)
    # Matched pair must have identical channel
    assert ws["f0_u_w0"].channel == ws["f1_u_w0"].channel
    assert ws["f0_u_w0"].channel is not None


def test_cross_face_segment_assignment():
    """face0 wire → segment=1; face1 wire → segment=0 (research wcpy-zv1 rule)."""
    face0, face1 = build_two_face_scenario()
    assign_vd_channels(face0, face1)
    ws = all_wires(face0, face1)
    assert ws["f0_u_w0"].segment == 1
    assert ws["f1_u_w0"].segment == 0


def test_standalone_wires_get_segment_zero():
    face0, face1 = build_two_face_scenario()
    assign_vd_channels(face0, face1)
    ws = all_wires(face0, face1)
    for name in ("f0_u_w1", "f1_u_w1", "f0_z_w0", "f1_z_w0"):
        assert ws[name].segment == 0, f"{name}.segment should be 0"


def test_all_wires_get_a_channel():
    face0, face1 = build_two_face_scenario()
    assign_vd_channels(face0, face1)
    ws = all_wires(face0, face1)
    for name, w in ws.items():
        assert w.channel is not None, f"{name}.channel should not be None"


def test_channel_count_equals_next_ch():
    """Number of distinct channels assigned equals the returned next_ch."""
    face0, face1 = build_two_face_scenario()
    next_ch = assign_vd_channels(face0, face1, start_channel=0)
    ws = all_wires(face0, face1)
    distinct = {w.channel for w in ws.values()}
    # 1 paired channel + 4 standalone channels = 5 unique; next_ch must agree
    assert len(distinct) == next_ch


def test_start_channel_offset():
    face0, face1 = build_two_face_scenario()
    assign_vd_channels(face0, face1, start_channel=100)
    ws = all_wires(face0, face1)
    for w in ws.values():
        assert w.channel >= 100


def test_returns_next_available_channel():
    face0, face1 = build_two_face_scenario()
    # 1 matched pair (→1 channel) + 4 standalone (→4 channels) = 5 channels total
    next_ch = assign_vd_channels(face0, face1, start_channel=0)
    assert next_ch == 5


def test_chaining_across_anodes():
    """next_ch from anode 0 becomes start_channel for anode 1."""
    face0a, face1a = build_two_face_scenario()
    face0b, face1b = build_two_face_scenario()
    next_ch = assign_vd_channels(face0a, face1a, start_channel=0)
    assign_vd_channels(face0b, face1b, start_channel=next_ch)
    # anode-b channels must all be >= next_ch
    for face in (face0b, face1b):
        for plane in face.planes:
            for w in plane.wires:
                assert w.channel >= next_ch


def test_no_overlap_between_anodes():
    """Channel numbers across two anodes must be disjoint."""
    face0a, face1a = build_two_face_scenario()
    face0b, face1b = build_two_face_scenario()
    next_ch = assign_vd_channels(face0a, face1a, start_channel=0)
    assign_vd_channels(face0b, face1b, start_channel=next_ch)
    ws_a = all_wires(face0a, face1a)
    ws_b = all_wires(face0b, face1b)
    channels_a = {w.channel for w in ws_a.values()}
    channels_b = {w.channel for w in ws_b.values()}
    assert channels_a.isdisjoint(channels_b)


def test_collection_wires_get_unique_channels_by_default():
    face0, face1 = build_two_face_scenario()
    assign_vd_channels(face0, face1)
    ws = all_wires(face0, face1)
    # Z-plane wires should each get their own channel
    assert ws["f0_z_w0"].channel != ws["f1_z_w0"].channel


def test_single_face_no_pairs():
    """One face with no partner: all wires get segment=0, unique channels."""
    face = FaceGeom(name="face0", planes=[
        PlaneGeom(name="volTPCPlaneU0", wires=[
            make_wire("w0", tail=(0, -100, 0), head=(0, 0, 100)),
            make_wire("w1", tail=(0, -90, 0),  head=(0, 0, 90)),
        ]),
    ])
    empty = FaceGeom(name="face1", planes=[
        PlaneGeom(name="volTPCPlaneU0", wires=[]),
    ])
    next_ch = assign_vd_channels(face, empty, start_channel=0)
    for plane in face.planes:
        for w in plane.wires:
            assert w.segment == 0
            assert w.channel is not None
    assert next_ch == 2
