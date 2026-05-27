"""Tests for WireGeom channel/segment fields in wirecell.util.gdml."""

from wirecell.util.gdml import WireGeom


def test_wiregeom_channel_defaults_none():
    w = WireGeom(name="w", tail=None, head=None, radius=0.15, plane_name="p")
    assert w.channel is None
    assert w.segment is None


def test_wiregeom_channel_can_be_set():
    w = WireGeom(name="w", tail=None, head=None, radius=0.15, plane_name="p",
                 channel=42, segment=1)
    assert w.channel == 42
    assert w.segment == 1
