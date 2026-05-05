"""
GDML geometry utilities for Wire-Cell.

This module provides tools for reading GDML XML geometry files and
converting them to the Wire-Cell wires schema.

GDML rotation convention
------------------------
A GDML ``<rotation x="a" y="b" z="c" unit="deg"/>`` stores a *passive*
rotation matrix built by applying extrinsic rotations in X-then-Y-then-Z order::

    M_passive = Mz(c) · My(b) · Mx(a)

The *active* local-to-world rotation needed to transform points is the inverse
(= transpose for orthogonal matrices)::

    R_active = M_passive^{-1} = M_passive^T

The full 4x4 homogeneous local-to-world matrix for a physvol is::

    T = [[R_active | t],
         [0  0  0  | 1]]

Hierarchy composition (parent first, child last)::

    T_world = T_cryostat @ T_tpc @ T_plane @ T_wire_physvol

Wire endpoint extraction
------------------------
A GDML ``<tube z="L" …/>`` stores the **full** length ``L``.
The wire axis is the local Z-axis; the two endpoints in local coordinates are::

    tail_local = [0, 0, -L/2]
    head_local = [0, 0, +L/2]

Apply the world transform to get world-frame endpoints.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def gdml_transform(pos_xyz_mm, rot_xyz_deg):
    """
    Build a 4x4 local-to-world homogeneous transform from a GDML physvol.

    Args:
        pos_xyz_mm:  sequence of 3 floats — translation (typically in mm)
        rot_xyz_deg: sequence of 3 floats — GDML rotation x/y/z in degrees

    Returns:
        (4, 4) float64 ndarray T such that ``p_world = T @ [*p_local, 1]``
    """
    # 'xyz' = extrinsic rotations about fixed X, then Y, then Z — the GDML convention.
    # .inv() converts the passive frame rotation to the active point transformation.
    R = Rotation.from_euler('xyz', rot_xyz_deg, degrees=True).inv().as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos_xyz_mm
    return T


def compose_transforms(*transforms):
    """
    Compose a sequence of 4x4 homogeneous transforms parent-first, child-last.

    Example::

        T_world = compose_transforms(T_cryo, T_tpc, T_plane, T_wire)

    Returns the identity matrix when called with no arguments.
    """
    result = np.eye(4)
    for T in transforms:
        result = result @ T
    return result


def apply_transform(T, point_local):
    """
    Apply a 4x4 homogeneous matrix to a 3-vector.

    Args:
        T:           (4, 4) ndarray from :func:`gdml_transform` or :func:`compose_transforms`
        point_local: sequence of 3 floats

    Returns:
        (3,) float64 ndarray — the point in the parent frame
    """
    p = np.asarray(point_local, dtype=float)
    return (T @ np.append(p, 1.0))[:3]
