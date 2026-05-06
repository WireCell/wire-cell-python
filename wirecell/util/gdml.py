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

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from scipy.spatial.transform import Rotation

# ── Detector config ────────────────────────────────────────────────────────────

_REQUIRED_CONFIG_KEYS = frozenset({"role_patterns", "connectivity_mode", "nearness_tolerance"})

# Named built-in configs.  Populated by wcpy-8z5 (protodunevd_v4, protodunevd_v5).
BUILTIN_CONFIGS: dict[str, dict] = {}


def load_config(path_or_name: Union[str, pathlib.Path]) -> dict:
    """Load a detector configuration dict from a JSON file or a built-in name.

    Args:
        path_or_name: Either a path to a JSON config file (``str`` or
            ``pathlib.Path``) or the name of a built-in detector config.

    Required JSON keys:

    * ``role_patterns`` — ``dict[str, str]``: maps role names
      (``"wire"``, ``"plane"``, ``"face"``, ``"detector"``) to regex strings
      that match the corresponding GDML logical-volume names.
    * ``connectivity_mode`` — ``"vd"`` or ``"hd"``.
    * ``nearness_tolerance`` — positive ``float``, in cm; used when matching
      wire endpoints across faces.

    Returns:
        The validated config dict (a fresh copy for file-based configs).

    Raises:
        ValueError: If the name is not a known built-in and not a valid file
            path, or if any required key is absent from the loaded config.
    """
    p = pathlib.Path(path_or_name)
    if p.exists():
        with p.open() as fh:
            cfg = json.load(fh)
    elif str(path_or_name) in BUILTIN_CONFIGS:
        cfg = dict(BUILTIN_CONFIGS[str(path_or_name)])
    else:
        known = sorted(BUILTIN_CONFIGS)
        raise ValueError(
            f"Unknown detector config {path_or_name!r}. "
            f"Provide a JSON file path or one of the built-in names: {known}"
        )

    missing = _REQUIRED_CONFIG_KEYS - cfg.keys()
    if missing:
        raise ValueError(
            f"Detector config is missing required key(s): {sorted(missing)}"
        )

    return cfg


@dataclass
class WireGeom:
    """Intermediate representation of a single wire in world-frame coordinates."""
    name: str
    tail: Optional[np.ndarray]
    head: Optional[np.ndarray]
    radius: float
    plane_name: str
    channel: Optional[int] = None
    segment: Optional[int] = None


@dataclass
class PlaneGeom:
    """Intermediate representation of a wire plane."""
    name: str
    wires: list[WireGeom] = field(default_factory=list)


@dataclass
class FaceGeom:
    """Intermediate representation of one TPC face (collection of planes)."""
    name: str
    planes: list[PlaneGeom] = field(default_factory=list)


@dataclass
class AnodeGeom:
    """Intermediate representation of an anode (one or two faces)."""
    faces: list[FaceGeom] = field(default_factory=list)


@dataclass
class DetectorGeom:
    """Intermediate representation of a full detector (collection of anodes)."""
    anodes: list[AnodeGeom] = field(default_factory=list)


def _endpoint_key(point: np.ndarray) -> tuple[int, int, int]:
    """Round a 3-vector to the nearest mm for spatial hashing."""
    return (int(round(point[0])), int(round(point[1])), int(round(point[2])))


def _is_collection_plane(plane: PlaneGeom) -> bool:
    """True if the plane name indicates a collection (Z or W) plane."""
    return 'Z' in plane.name or 'W' in plane.name


def _match_plane_wires(
    plane0: PlaneGeom,
    plane1: PlaneGeom,
) -> list[tuple[WireGeom, WireGeom]]:
    """Return (wire_from_plane0, wire_from_plane1) pairs that share an endpoint.

    Matching uses a 1 mm grid hash: two endpoints whose coordinates all round to
    the same integer mm values are considered coincident.  This handles floating-
    point noise in GDML world-frame coordinates (GDML endpoints either coincide
    exactly or are separated by at least the wire pitch, >> 1 mm).
    Each wire from plane0 appears in at most one returned pair.
    """
    # Build endpoint → wire lookup for plane0
    endpoint_map: dict[tuple[int, int, int], WireGeom] = {}
    for wire in plane0.wires:
        for pt in (wire.tail, wire.head):
            if pt is not None:
                endpoint_map[_endpoint_key(pt)] = wire

    pairs: list[tuple[WireGeom, WireGeom]] = []
    matched_wire0_names: set[str] = set()
    for wire1 in plane1.wires:
        for pt in (wire1.tail, wire1.head):
            if pt is None:
                continue
            wire0 = endpoint_map.get(_endpoint_key(pt))
            if wire0 is not None and wire0.name not in matched_wire0_names:
                pairs.append((wire0, wire1))
                matched_wire0_names.add(wire0.name)
                break
    return pairs


def find_cross_face_pairs(
    face0: FaceGeom,
    face1: FaceGeom,
    skip_collection: bool = True,
) -> list[tuple[WireGeom, WireGeom]]:
    """Find cross-face connected wire pairs between two faces of the same anode.

    For each corresponding plane pair (matched by position in the ``planes``
    list), finds wires from ``face0`` and ``face1`` that share a world-frame
    endpoint.  Collection planes (names containing ``'Z'`` or ``'W'``) are
    skipped when ``skip_collection=True`` (the default), since VD collection
    wires are never connected across faces.

    Returns a list of ``(wire_from_face0, wire_from_face1)`` tuples.
    """
    pairs: list[tuple[WireGeom, WireGeom]] = []
    for plane0, plane1 in zip(face0.planes, face1.planes):
        if skip_collection and _is_collection_plane(plane0):
            continue
        pairs.extend(_match_plane_wires(plane0, plane1))
    return pairs


def assign_vd_channels(
    face0: FaceGeom,
    face1: FaceGeom,
    start_channel: int = 0,
    skip_collection: bool = True,
) -> int:
    """Assign channel and segment numbers to wires across two VD anode faces.

    Populates ``WireGeom.channel`` and ``WireGeom.segment`` in-place for every
    wire in ``face0`` and ``face1``.

    Rules (from research wcpy-zv1):

    * **Cross-face connected pair** — found by :func:`find_cross_face_pairs`:
      both wires share one channel number; the ``face1`` (index-1, higher ident)
      wire gets ``segment=0``; the ``face0`` (index-0, lower ident) wire gets
      ``segment=1``.
    * **Standalone wire** (no cross-face match): ``segment=0``, unique channel.

    Collection planes (``skip_collection=True``) are not matched across faces so
    all their wires are treated as standalone.

    Args:
        face0: First face (lower ident, ``segment=1`` for cross-face wires).
        face1: Second face (higher ident, ``segment=0`` for cross-face wires).
        start_channel: First channel number to assign (enables global uniqueness
            when chaining across anodes).
        skip_collection: If True (default), collection planes are not matched.

    Returns:
        The next available channel number (= ``start_channel`` + total channels
        assigned).
    """
    ch = start_channel

    # Identify cross-face connected pairs.
    pairs = find_cross_face_pairs(face0, face1, skip_collection=skip_collection)

    # Track which wires are in a pair (by name, since np.ndarray prevents hashing).
    matched0: dict[str, int] = {}  # wire.name → assigned channel
    matched1: dict[str, int] = {}

    for wire0, wire1 in pairs:
        wire0.channel = ch
        wire0.segment = 1
        wire1.channel = ch
        wire1.segment = 0
        matched0[wire0.name] = ch
        matched1[wire1.name] = ch
        ch += 1

    # Assign standalone channels to all unmatched wires in both faces.
    for face, matched in ((face0, matched0), (face1, matched1)):
        for plane in face.planes:
            for wire in plane.wires:
                if wire.name not in matched:
                    wire.channel = ch
                    wire.segment = 0
                    ch += 1

    return ch


def match_role(name: str, patterns: dict) -> Optional[str]:
    """Return the first role whose regex pattern fully matches *name*, or None.

    Args:
        name:     A GDML logical volume name.
        patterns: ``dict[str, str]`` mapping role names to regex strings
                  (as returned by the ``role_patterns`` key of a detector
                  config loaded with :func:`load_config`).

    Returns:
        The matching role name (e.g. ``"wire"``, ``"plane"``), or ``None`` if
        no pattern matches.
    """
    for role, pattern in patterns.items():
        if re.fullmatch(pattern, name):
            return role
    return None


def classify_volumes(vol_names, patterns: dict) -> dict:
    """Partition a collection of volume names into roles.

    Args:
        vol_names: Iterable of GDML logical volume name strings.
        patterns:  ``dict[str, str]`` role→regex map (see :func:`match_role`).

    Returns:
        ``dict[str, list[str]]`` mapping each role to the list of volume names
        that matched it.  Volumes that match no pattern are silently omitted.
        The order of names within each role list matches the input order.
    """
    result: dict[str, list] = {}
    for name in vol_names:
        role = match_role(name, patterns)
        if role is not None:
            result.setdefault(role, []).append(name)
    return result


def parse_define(gdml_root) -> dict:
    """Parse the GDML <define> section into named positions and rotations.

    Args:
        gdml_root: The root Element of a parsed GDML document.

    Returns:
        A dict with two keys:

        * ``"positions"`` — ``dict[str, np.ndarray]``: named position vectors
          in mm (shape ``(3,)``).
        * ``"rotations"`` — ``dict[str, np.ndarray]``: named rotation Euler
          angles in **radians** (shape ``(3,)``), ordered ``(rx, ry, rz)`` in
          the GDML extrinsic X→Y→Z convention.
    """
    _UNIT_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0}
    _UNIT_TO_RAD = {"deg": np.pi / 180.0, "rad": 1.0}

    positions: dict[str, np.ndarray] = {}
    rotations: dict[str, np.ndarray] = {}

    define = gdml_root.find("define")
    if define is None:
        return {"positions": positions, "rotations": rotations}

    for elem in define:
        name = elem.get("name")
        if name is None:
            continue
        if elem.tag == "position":
            unit = elem.get("unit", "mm")
            scale = _UNIT_TO_MM.get(unit, 1.0)
            x = float(elem.get("x", 0.0)) * scale
            y = float(elem.get("y", 0.0)) * scale
            z = float(elem.get("z", 0.0)) * scale
            positions[name] = np.array([x, y, z], dtype=float)
        elif elem.tag == "rotation":
            unit = elem.get("unit", "deg")
            scale = _UNIT_TO_RAD.get(unit, np.pi / 180.0)
            rx = float(elem.get("x", 0.0)) * scale
            ry = float(elem.get("y", 0.0)) * scale
            rz = float(elem.get("z", 0.0)) * scale
            rotations[name] = np.array([rx, ry, rz], dtype=float)

    return {"positions": positions, "rotations": rotations}


def parse_solids(gdml_root) -> dict:
    """Parse the GDML <solids> section and return tube dimensions.

    Only ``<tube>`` elements are returned; box and other solid types are
    ignored.

    Args:
        gdml_root: The root Element of a parsed GDML document.

    Returns:
        ``dict[str, dict]`` mapping each tube solid name to::

            {"rmax": float,   # outer radius in mm
             "half_z": float} # half-length in mm (GDML z attribute / 2)
    """
    _UNIT_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0}

    tubes: dict[str, dict] = {}

    solids = gdml_root.find("solids")
    if solids is None:
        return tubes

    for elem in solids:
        if elem.tag != "tube":
            continue
        name = elem.get("name")
        if name is None:
            continue
        unit = elem.get("lunit", "mm")
        scale = _UNIT_TO_MM.get(unit, 1.0)
        rmax = float(elem.get("rmax", 0.0)) * scale
        full_z = float(elem.get("z", 0.0)) * scale
        tubes[name] = {"rmax": rmax, "half_z": full_z / 2.0}

    return tubes


def parse_structure(gdml_root, defines: dict, solids: dict) -> dict:
    """Parse the GDML <structure> section into a logical volume tree.

    Args:
        gdml_root: Root Element of a parsed GDML document.
        defines:   Output of :func:`parse_define` — supplies named position
                   and rotation lookups.
        solids:    Output of :func:`parse_solids` — currently unused but
                   accepted for a consistent call signature.

    Returns:
        ``dict[str, dict]`` mapping each logical volume name to::

            {
                "solid":    str,          # solidref name
                "physvols": [             # child placements, in order
                    {
                        "name": str,      # physvol name (or volumeref if anonymous)
                        "vol":  str,      # child logical volume name
                        "pos":  ndarray,  # translation in mm, shape (3,)
                        "rot":  ndarray,  # Euler angles in radians (rx,ry,rz), shape (3,)
                    },
                    ...
                ],
            }
    """
    _UNIT_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0}
    _UNIT_TO_RAD = {"deg": np.pi / 180.0, "rad": 1.0}

    def _pos_from_elem(elem) -> np.ndarray:
        unit = elem.get("unit", "mm")
        scale = _UNIT_TO_MM.get(unit, 1.0)
        return np.array(
            [float(elem.get("x", 0.0)) * scale,
             float(elem.get("y", 0.0)) * scale,
             float(elem.get("z", 0.0)) * scale],
            dtype=float,
        )

    def _rot_from_elem(elem) -> np.ndarray:
        unit = elem.get("unit", "deg")
        scale = _UNIT_TO_RAD.get(unit, np.pi / 180.0)
        return np.array(
            [float(elem.get("x", 0.0)) * scale,
             float(elem.get("y", 0.0)) * scale,
             float(elem.get("z", 0.0)) * scale],
            dtype=float,
        )

    positions = defines.get("positions", {})
    rotations = defines.get("rotations", {})
    _zero3 = np.zeros(3, dtype=float)

    volumes: dict = {}

    structure = gdml_root.find("structure")
    if structure is None:
        return volumes

    for vol_elem in structure.findall("volume"):
        vol_name = vol_elem.get("name")
        if vol_name is None:
            continue

        solid_ref = ""
        solidref_elem = vol_elem.find("solidref")
        if solidref_elem is not None:
            solid_ref = solidref_elem.get("ref", "")

        physvol_list = []
        for pv_elem in vol_elem.findall("physvol"):
            volref_elem = pv_elem.find("volumeref")
            if volref_elem is None:
                continue
            child_vol = volref_elem.get("ref", "")

            pv_name = pv_elem.get("name") or child_vol

            # Resolve position
            posref = pv_elem.find("positionref")
            pos_inline = pv_elem.find("position")
            if posref is not None:
                pos = positions.get(posref.get("ref", ""), _zero3.copy())
            elif pos_inline is not None:
                pos = _pos_from_elem(pos_inline)
            else:
                pos = _zero3.copy()

            # Resolve rotation
            rotref = pv_elem.find("rotationref")
            rot_inline = pv_elem.find("rotation")
            if rotref is not None:
                rot = rotations.get(rotref.get("ref", ""), _zero3.copy())
            elif rot_inline is not None:
                rot = _rot_from_elem(rot_inline)
            else:
                rot = _zero3.copy()

            physvol_list.append({
                "name": pv_name,
                "vol": child_vol,
                "pos": pos,
                "rot": rot,
            })

        volumes[vol_name] = {"solid": solid_ref, "physvols": physvol_list}

    return volumes


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
