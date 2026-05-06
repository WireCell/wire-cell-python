"""Tests for cross-face wire endpoint matching in wirecell.util.gdml."""

import numpy as np
import pytest
from wirecell.util.gdml import (
    WireGeom, PlaneGeom, FaceGeom,
    find_cross_face_pairs,
)


def make_wire(name, tail, head, plane_name="volTPCPlaneU0"):
    return WireGeom(name=name, tail=np.array(tail, float),
                    head=np.array(head, float), radius=0.15,
                    plane_name=plane_name)


def pair_names(pairs):
    """Return sorted set of (name0, name1) from matched pairs."""
    return sorted((w0.name, w1.name) for w0, w1 in pairs)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def two_face_induction_match():
    """
    Two U-plane wires per face.  The junction is at y=0.
      face0: wires end (head) at the junction.
      face1: wires start (tail) at the junction.
    All four wires should be paired 2→2.
    """
    junction = [(0, 0, 100), (0, 0, 90)]
    face0_u = PlaneGeom(name="volTPCPlaneU0", wires=[
        make_wire("f0_u_w0", tail=(0, -100, 0),   head=junction[0]),
        make_wire("f0_u_w1", tail=(0, -90,  0),   head=junction[1]),
    ])
    face1_u = PlaneGeom(name="volTPCPlaneU0", wires=[
        make_wire("f1_u_w0", tail=junction[0], head=(0, 100, 200)),
        make_wire("f1_u_w1", tail=junction[1], head=(0, 100, 190)),
    ])
    face0 = FaceGeom(name="face0", planes=[face0_u])
    face1 = FaceGeom(name="face1", planes=[face1_u])
    return face0, face1


def two_face_with_collection():
    """
    U-plane with matching wires + Z-plane (collection) with matching wires.
    """
    junction = (0, 0, 100)
    face0_u = PlaneGeom(name="volTPCPlaneU0", wires=[
        make_wire("f0_u_w0", tail=(0, -100, 0), head=junction),
    ])
    face1_u = PlaneGeom(name="volTPCPlaneU0", wires=[
        make_wire("f1_u_w0", tail=junction, head=(0, 100, 200)),
    ])
    face0_z = PlaneGeom(name="volTPCPlaneZ0", wires=[
        make_wire("f0_z_w0", tail=(0, -100, 50), head=(0, 0, 50), plane_name="volTPCPlaneZ0"),
    ])
    face1_z = PlaneGeom(name="volTPCPlaneZ0", wires=[
        make_wire("f1_z_w0", tail=(0, 0, 50), head=(0, 100, 50), plane_name="volTPCPlaneZ0"),
    ])
    face0 = FaceGeom(name="face0", planes=[face0_u, face0_z])
    face1 = FaceGeom(name="face1", planes=[face1_u, face1_z])
    return face0, face1


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_basic_match_via_head_and_tail():
    face0, face1 = two_face_induction_match()
    pairs = find_cross_face_pairs(face0, face1)
    assert len(pairs) == 2
    names = pair_names(pairs)
    assert ("f0_u_w0", "f1_u_w0") in names
    assert ("f0_u_w1", "f1_u_w1") in names


def test_no_shared_endpoints_returns_empty():
    face0_u = PlaneGeom(name="volTPCPlaneU0", wires=[
        make_wire("f0_w0", tail=(0, -100, 0), head=(0, 0, 100)),
    ])
    face1_u = PlaneGeom(name="volTPCPlaneU0", wires=[
        make_wire("f1_w0", tail=(0, 50, 300), head=(0, 100, 400)),
    ])
    face0 = FaceGeom(name="face0", planes=[face0_u])
    face1 = FaceGeom(name="face1", planes=[face1_u])
    pairs = find_cross_face_pairs(face0, face1)
    assert pairs == []


def test_partial_match():
    """Only one of two wire pairs shares an endpoint."""
    junction = (0, 0, 100)
    face0_u = PlaneGeom(name="volTPCPlaneU0", wires=[
        make_wire("f0_w0", tail=(0, -100, 0),   head=junction),  # matches
        make_wire("f0_w1", tail=(0, -90,  0),   head=(0, 0, 80)),  # no match
    ])
    face1_u = PlaneGeom(name="volTPCPlaneU0", wires=[
        make_wire("f1_w0", tail=junction,      head=(0, 100, 200)),  # matches f0_w0
        make_wire("f1_w1", tail=(0, 50, 999),  head=(0, 100, 999)),  # no match
    ])
    face0 = FaceGeom(name="face0", planes=[face0_u])
    face1 = FaceGeom(name="face1", planes=[face1_u])
    pairs = find_cross_face_pairs(face0, face1)
    assert len(pairs) == 1
    assert pair_names(pairs) == [("f0_w0", "f1_w0")]


def test_match_via_tail_of_face0_wire():
    """The shared endpoint can be face0's tail (not just head)."""
    junction = (0, 0, 0)
    face0_u = PlaneGeom(name="volTPCPlaneU0", wires=[
        make_wire("f0_w0", tail=junction, head=(0, 100, 100)),  # tail at junction
    ])
    face1_u = PlaneGeom(name="volTPCPlaneU0", wires=[
        make_wire("f1_w0", tail=(0, -100, -100), head=junction),  # head at junction
    ])
    face0 = FaceGeom(name="face0", planes=[face0_u])
    face1 = FaceGeom(name="face1", planes=[face1_u])
    pairs = find_cross_face_pairs(face0, face1)
    assert len(pairs) == 1
    assert pair_names(pairs) == [("f0_w0", "f1_w0")]


def test_collection_plane_skipped_by_default():
    face0, face1 = two_face_with_collection()
    pairs = find_cross_face_pairs(face0, face1)
    names = pair_names(pairs)
    assert ("f0_u_w0", "f1_u_w0") in names
    # collection wire should NOT appear
    assert all("z" not in n0.lower() and "z" not in n1.lower() for n0, n1 in names)


def test_collection_plane_included_when_skip_false():
    face0, face1 = two_face_with_collection()
    pairs = find_cross_face_pairs(face0, face1, skip_collection=False)
    names = pair_names(pairs)
    assert ("f0_u_w0", "f1_u_w0") in names
    assert ("f0_z_w0", "f1_z_w0") in names


def test_w_plane_skipped_by_name():
    """Plane named with 'W' is also treated as collection and skipped."""
    junction = (0, 0, 100)
    face0_w = PlaneGeom(name="volTPCPlaneW0", wires=[
        make_wire("f0_w0", tail=(0, -100, 0), head=junction, plane_name="volTPCPlaneW0"),
    ])
    face1_w = PlaneGeom(name="volTPCPlaneW0", wires=[
        make_wire("f1_w0", tail=junction, head=(0, 100, 200), plane_name="volTPCPlaneW0"),
    ])
    face0 = FaceGeom(name="face0", planes=[face0_w])
    face1 = FaceGeom(name="face1", planes=[face1_w])
    pairs = find_cross_face_pairs(face0, face1)
    assert pairs == []


def test_none_endpoint_does_not_crash():
    """Wire with None tail/head is silently skipped during matching."""
    junction = (0, 0, 100)
    face0_u = PlaneGeom(name="volTPCPlaneU0", wires=[
        WireGeom(name="f0_w0", tail=None, head=np.array(junction, float),
                 radius=0.15, plane_name="volTPCPlaneU0"),
    ])
    face1_u = PlaneGeom(name="volTPCPlaneU0", wires=[
        WireGeom(name="f1_w0", tail=np.array(junction, float), head=None,
                 radius=0.15, plane_name="volTPCPlaneU0"),
    ])
    face0 = FaceGeom(name="face0", planes=[face0_u])
    face1 = FaceGeom(name="face1", planes=[face1_u])
    pairs = find_cross_face_pairs(face0, face1)
    assert len(pairs) == 1
    assert pair_names(pairs) == [("f0_w0", "f1_w0")]


def test_floating_point_noise_within_half_mm_matches():
    """Endpoints within 0.4 mm of each other (sub-mm noise) are matched."""
    pt0 = (0.0, 0.0, 100.0)
    pt1 = (0.0, 0.3, 100.2)  # < 0.5 mm away → rounds to same mm grid point
    face0_u = PlaneGeom(name="volTPCPlaneU0", wires=[
        make_wire("f0_w0", tail=(0, -100, 0), head=pt0),
    ])
    face1_u = PlaneGeom(name="volTPCPlaneU0", wires=[
        make_wire("f1_w0", tail=pt1, head=(0, 100, 200)),
    ])
    face0 = FaceGeom(name="face0", planes=[face0_u])
    face1 = FaceGeom(name="face1", planes=[face1_u])
    pairs = find_cross_face_pairs(face0, face1)
    assert len(pairs) == 1


def test_empty_planes_return_empty():
    face0 = FaceGeom(name="face0", planes=[PlaneGeom(name="volTPCPlaneU0", wires=[])])
    face1 = FaceGeom(name="face1", planes=[PlaneGeom(name="volTPCPlaneU0", wires=[])])
    pairs = find_cross_face_pairs(face0, face1)
    assert pairs == []


def test_mismatched_plane_counts_uses_zip():
    """If face0 has more planes than face1, extra planes are silently ignored."""
    junction = (0, 0, 100)
    u0 = PlaneGeom(name="volTPCPlaneU0", wires=[
        make_wire("f0_w0", tail=(0, -100, 0), head=junction),
    ])
    u1 = PlaneGeom(name="volTPCPlaneU0", wires=[
        make_wire("f1_w0", tail=junction, head=(0, 100, 200)),
    ])
    extra = PlaneGeom(name="volTPCPlaneV0", wires=[
        make_wire("f0_v_w0", tail=(0, -100, 50), head=(0, 0, 150)),
    ])
    face0 = FaceGeom(name="face0", planes=[u0, extra])  # 2 planes
    face1 = FaceGeom(name="face1", planes=[u1])           # 1 plane
    pairs = find_cross_face_pairs(face0, face1)
    assert len(pairs) == 1  # only the U match; extra plane silently skipped
