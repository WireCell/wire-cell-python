#!/usr/bin/env python
"""
Tests for GDML coordinate transform utilities in wirecell.util.gdml.

GDML rotation convention:
  <rotation x="a" y="b" z="c"/> stores a passive rotation M = Mz(c)*My(b)*Mx(a).
  The active local-to-world rotation is M^{-1} = M^T.
  gdml_transform returns the 4x4 matrix T such that p_world = T @ [p_local, 1].
"""
import numpy as np
import pytest
from wirecell.util.gdml import gdml_transform, compose_transforms, apply_transform


def test_identity_transform():
    T = gdml_transform([0, 0, 0], [0, 0, 0])
    assert T.shape == (4, 4)
    np.testing.assert_allclose(T, np.eye(4), atol=1e-12)


def test_apply_identity():
    T = gdml_transform([0, 0, 0], [0, 0, 0])
    p = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(apply_transform(T, p), p, atol=1e-12)


def test_pure_translation():
    T = gdml_transform([10, 20, 30], [0, 0, 0])
    p = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(apply_transform(T, p), [11.0, 22.0, 33.0], atol=1e-12)


def test_rotation_x90_z_to_y():
    # x=90 deg: local Z-axis should map to world +Y
    T = gdml_transform([0, 0, 0], [90, 0, 0])
    np.testing.assert_allclose(apply_transform(T, [0, 0, 1]), [0, 1, 0], atol=1e-12)


def test_rotation_x90_y_to_neg_z():
    # x=90 deg: local Y-axis should map to world -Z
    T = gdml_transform([0, 0, 0], [90, 0, 0])
    np.testing.assert_allclose(apply_transform(T, [0, 1, 0]), [0, 0, -1], atol=1e-12)


def test_rotation_x90_x_unchanged():
    # x=90 deg: local X-axis must not move
    T = gdml_transform([0, 0, 0], [90, 0, 0])
    np.testing.assert_allclose(apply_transform(T, [1, 0, 0]), [1, 0, 0], atol=1e-12)


def test_rUWireAboutX_30deg_from_Z():
    # rUWireAboutX from protodunevd GDML: x=150, y=0, z=0
    # Wire tube axis (local Z) must make a 30-degree angle with the world Z-axis.
    T = gdml_transform([0, 0, 0], [150, 0, 0])
    wire_dir = apply_transform(T, [0, 0, 1]) - apply_transform(T, [0, 0, 0])
    cos_angle = abs(np.dot(wire_dir / np.linalg.norm(wire_dir), [0, 0, 1]))
    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    np.testing.assert_allclose(angle_deg, 30.0, atol=1e-10)


def test_transform_rotation_matrix_orthogonal():
    # The 3x3 rotation block must be orthogonal for arbitrary angles
    T = gdml_transform([5, -3, 7], [45, 30, 60])
    assert T.shape == (4, 4)
    np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=1e-12)
    R = T[:3, :3]
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)


def test_two_level_composition():
    # Level 1 (parent): translate 100 along X, no rotation
    T1 = gdml_transform([100, 0, 0], [0, 0, 0])
    # Level 2 (child): rotate 90 deg about X (local Z -> world Y), no translation
    T2 = gdml_transform([0, 0, 0], [90, 0, 0])
    T_world = compose_transforms(T1, T2)
    # Local [0, 0, 5]: rotation maps Z->Y giving [0,5,0], translation gives [100,5,0]
    np.testing.assert_allclose(apply_transform(T_world, [0, 0, 5]), [100, 5, 0], atol=1e-12)


def test_compose_three_levels():
    # Chain: translate X, then translate Y, then translate Z
    T1 = gdml_transform([1, 0, 0], [0, 0, 0])
    T2 = gdml_transform([0, 2, 0], [0, 0, 0])
    T3 = gdml_transform([0, 0, 3], [0, 0, 0])
    T = compose_transforms(T1, T2, T3)
    np.testing.assert_allclose(apply_transform(T, [0, 0, 0]), [1, 2, 3], atol=1e-12)


def test_compose_empty_is_identity():
    T = compose_transforms()
    np.testing.assert_allclose(T, np.eye(4), atol=1e-12)


def test_compose_single_is_passthrough():
    T0 = gdml_transform([5, 6, 7], [30, 45, 60])
    T = compose_transforms(T0)
    np.testing.assert_allclose(T, T0, atol=1e-12)


def test_wire_endpoints_from_gdml_tube_z():
    # GDML <tube z="10.0" .../> stores the FULL length; half_L = z_gdml / 2 = 5.
    # Wire tube axis is local Z; identity transform keeps them in local frame.
    z_gdml = 10.0
    half_L = z_gdml / 2
    T = gdml_transform([0, 0, 0], [0, 0, 0])
    tail = apply_transform(T, [0, 0, -half_L])
    head = apply_transform(T, [0, 0,  half_L])
    np.testing.assert_allclose(tail, [0, 0, -5.0], atol=1e-12)
    np.testing.assert_allclose(head, [0, 0,  5.0], atol=1e-12)


def test_wire_endpoints_with_rotation_and_translation():
    # Wire at position (10, 20, 30), rotated x=90 (local Z -> world Y).
    # Tube full length = 6, half_L = 3.
    # Local tail = [0, 0, -3], head = [0, 0, +3].
    # After rotation: tail=[0,-3,0], head=[0,+3,0].
    # After translation: tail=[10,17,30], head=[10,23,30].
    T = gdml_transform([10, 20, 30], [90, 0, 0])
    half_L = 3.0
    tail = apply_transform(T, [0, 0, -half_L])
    head = apply_transform(T, [0, 0,  half_L])
    np.testing.assert_allclose(tail, [10, 17, 30], atol=1e-12)
    np.testing.assert_allclose(head, [10, 23, 30], atol=1e-12)
