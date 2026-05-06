"""Tests for intermediate geometry dataclasses in wirecell.util.gdml."""

import numpy as np
import pytest
from wirecell.util.gdml import WireGeom, PlaneGeom, FaceGeom, AnodeGeom, DetectorGeom


def make_wire(name="wire0", plane_name="planeU0", tail=None, head=None, radius=0.15):
    if tail is None:
        tail = np.array([0.0, 0.0, -100.0])
    if head is None:
        head = np.array([0.0, 0.0, 100.0])
    return WireGeom(name=name, tail=tail, head=head, radius=radius, plane_name=plane_name)


def test_wiregeom_construction():
    tail = np.array([1.0, 2.0, 3.0])
    head = np.array([4.0, 5.0, 6.0])
    w = WireGeom(name="wire0", tail=tail, head=head, radius=0.15, plane_name="planeU0")
    assert w.name == "wire0"
    assert w.plane_name == "planeU0"
    assert w.radius == pytest.approx(0.15)
    np.testing.assert_array_equal(w.tail, tail)
    np.testing.assert_array_equal(w.head, head)


def test_wiregeom_tail_none():
    # tail=None and head=None must not crash (transform is deferred)
    w = WireGeom(name="wire0", tail=None, head=None, radius=0.15, plane_name="planeU0")
    assert w.tail is None
    assert w.head is None


def test_wiregeom_head_none_tail_set():
    tail = np.array([0.0, 0.0, -50.0])
    w = WireGeom(name="wire1", tail=tail, head=None, radius=0.15, plane_name="planeV0")
    np.testing.assert_array_equal(w.tail, tail)
    assert w.head is None


def test_planegeom_construction():
    wires = [make_wire(f"wire{i}", plane_name="planeU0") for i in range(3)]
    p = PlaneGeom(name="planeU0", wires=wires)
    assert p.name == "planeU0"
    assert len(p.wires) == 3
    assert p.wires[1].name == "wire1"


def test_planegeom_empty_wires():
    p = PlaneGeom(name="planeU0", wires=[])
    assert p.wires == []


def test_facegeom_construction():
    planes = [
        PlaneGeom(name=f"plane{x}0", wires=[make_wire(plane_name=f"plane{x}0")])
        for x in ("U", "V", "W")
    ]
    f = FaceGeom(name="face0", planes=planes)
    assert f.name == "face0"
    assert len(f.planes) == 3
    assert f.planes[0].name == "planeU0"


def test_anodegeom_construction():
    def make_face(name):
        planes = [PlaneGeom(name=f"plane{x}{name[-1]}", wires=[]) for x in ("U", "V", "W")]
        return FaceGeom(name=name, planes=planes)

    anode = AnodeGeom(faces=[make_face("face0"), make_face("face1")])
    assert len(anode.faces) == 2
    assert anode.faces[0].name == "face0"


def test_detectorgeom_construction():
    def make_anode(idx):
        faces = [FaceGeom(name=f"face{idx}{j}", planes=[]) for j in range(2)]
        return AnodeGeom(faces=faces)

    det = DetectorGeom(anodes=[make_anode(0), make_anode(1)])
    assert len(det.anodes) == 2
    assert det.anodes[1].faces[0].name == "face10"


def test_detectorgeom_single_anode():
    anode = AnodeGeom(faces=[])
    det = DetectorGeom(anodes=[anode])
    assert len(det.anodes) == 1
    assert det.anodes[0].faces == []
