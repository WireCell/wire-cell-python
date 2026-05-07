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
    face_name: str = ""
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


def find_vd_connected_pairs(anode_geom, nearness_tolerance: float) -> dict:
    """Identify cross-face wire connectivity for a VD anode.

    Matches wires between the two faces of *anode_geom* by comparing all
    endpoint pairs.  Two wires are considered connected when at least one
    endpoint from each lies within *nearness_tolerance* mm of each other.
    Collection planes (names containing ``'Z'`` or ``'W'``) are not matched.

    Segment convention (from wcpy-zv1 research):

    * **face0** (``anode_geom.faces[0]``) connected wires → ``segment = 1``
    * **face1** (``anode_geom.faces[1]``) connected wires → ``segment = 0``
    * Collection and standalone wires → ``segment = 0``

    Args:
        anode_geom:          An :class:`AnodeGeom` with exactly two faces.
        nearness_tolerance:  Maximum distance in mm for two endpoints to be
                             considered coincident.

    Returns:
        ``dict[str, dict]`` keyed by ``wire.name`` (assumed unique within the
        anode).  Each value is::

            {"segment": int, "connected_to": str or None}

    Raises:
        ValueError: If *anode_geom* does not contain exactly two faces.
    """
    if len(anode_geom.faces) != 2:
        raise ValueError(
            f"VD anode must have exactly 2 faces; got {len(anode_geom.faces)}."
        )

    face0, face1 = anode_geom.faces
    result: dict = {}

    def _close(w0, w1) -> bool:
        for pt0 in (w0.tail, w0.head):
            for pt1 in (w1.tail, w1.head):
                if pt0 is not None and pt1 is not None:
                    if np.linalg.norm(pt0 - pt1) <= nearness_tolerance:
                        return True
        return False

    matched0: set = set()
    matched1: set = set()

    for plane0, plane1 in zip(face0.planes, face1.planes):
        if _is_collection_plane(plane0):
            continue
        for wire1 in plane1.wires:
            for wire0 in plane0.wires:
                if wire0.name in matched0:
                    continue
                if _close(wire0, wire1):
                    result[wire0.name] = {"segment": 1, "connected_to": wire1.name}
                    result[wire1.name] = {"segment": 0, "connected_to": wire0.name}
                    matched0.add(wire0.name)
                    matched1.add(wire1.name)
                    break

    # Standalone wires (unmatched or in collection planes)
    for face in (face0, face1):
        for plane in face.planes:
            for wire in plane.wires:
                if wire.name not in result:
                    result[wire.name] = {"segment": 0, "connected_to": None}

    return result


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


def extract_wires(
    vol_tree: dict,
    defines: dict,
    solids: dict,
    root_vol: str,
    patterns: dict,
) -> list:
    """Traverse a GDML logical volume tree and extract wire endpoints in world frame.

    Walks the hierarchy starting at *root_vol*.  When a volume whose name
    matches role ``"wire"`` is encountered its two endpoints are computed by
    applying the accumulated local-to-world transform to the canonical local
    tube endpoints ``[0, 0, ±half_z]``.  A volume whose name matches role
    ``"plane"`` sets the ``plane_name`` attribute carried down to its wire
    descendants.

    Args:
        vol_tree: Output of :func:`parse_structure`.
        defines:  Output of :func:`parse_define` (accepted for API symmetry;
                  transforms are already embedded in *vol_tree*).
        solids:   Output of :func:`parse_solids` — provides ``rmax`` and
                  ``half_z`` for each tube solid.
        root_vol: Name of the logical volume to start traversal from
                  (e.g. ``"volWorld"``).
        patterns: ``dict[str, str]`` role→regex map as returned by the
                  ``role_patterns`` key of a detector config.

    Returns:
        List of :class:`WireGeom` objects with world-frame ``tail`` and
        ``head`` endpoints, ``radius``, and ``plane_name`` populated.
        ``channel`` and ``segment`` are left ``None`` (filled later by
        :func:`assign_vd_channels`).
    """
    result = []

    def _recurse(vol_name, pv_name, transform_chain, plane_name, face_name):
        entry = vol_tree.get(vol_name)
        if entry is None:
            return

        role = match_role(vol_name, patterns)

        if role == "wire":
            solid_name = entry.get("solid", "")
            dims = solids.get(solid_name)
            if dims is not None:
                T = compose_transforms(*transform_chain)
                tail = apply_transform(T, [0.0, 0.0, -dims["half_z"]])
                head = apply_transform(T, [0.0, 0.0, +dims["half_z"]])
                result.append(WireGeom(
                    name=vol_name,
                    tail=tail,
                    head=head,
                    radius=dims["rmax"],
                    plane_name=plane_name,
                    face_name=face_name,
                ))
            return  # leaf — do not recurse into wire volumes

        if role == "face":
            # Use the physvol placement name to distinguish multiple instances
            # of the same face logical volume (e.g. Top vs Bot in VD open-book).
            face_name = pv_name
        elif role == "plane":
            plane_name = vol_name

        for pv in entry.get("physvols", []):
            T_pv = gdml_transform(pv["pos"], np.degrees(pv["rot"]))
            _recurse(pv["vol"], pv["name"], transform_chain + [T_pv], plane_name, face_name)

    _recurse(root_vol, root_vol, [], "", "")
    return result


def build_detector_faces(vol_tree: dict, wires_list: list, patterns: dict) -> list:
    """Group WireGeom objects into a PlaneGeom/FaceGeom hierarchy.

    Uses :attr:`WireGeom.face_name` (the physvol placement name set by
    :func:`extract_wires`) and :attr:`WireGeom.plane_name` to partition wires
    into faces and planes.  When ``face_name`` is empty (e.g. traversal started
    below the face level), the face LV name is looked up via the *vol_tree*
    parent chain as a fallback.

    Args:
        vol_tree: Output of :func:`parse_structure`.
        wires_list: Flat list of :class:`WireGeom` from :func:`extract_wires`.
        patterns:  ``dict[str, str]`` role→regex map from the detector config.

    Returns:
        List of :class:`FaceGeom` objects, each containing
        :class:`PlaneGeom` children whose ``wires`` lists are slices of
        *wires_list* (no copies made).
    """
    # Build LV child→parent map for face_name fallback.
    lv_parent: dict[str, str] = {}
    for vol_name, entry in vol_tree.items():
        for pv in entry.get("physvols", []):
            lv_parent[pv["vol"]] = vol_name

    def _face_lv(vol_name: str) -> str:
        visited: set = set()
        current = lv_parent.get(vol_name)
        while current and current not in visited:
            visited.add(current)
            if match_role(current, patterns) == "face":
                return current
            current = lv_parent.get(current)
        return ""

    face_plane_wires: dict = {}  # face_id → {plane_name → [WireGeom]}
    for wire in wires_list:
        face_id = wire.face_name if wire.face_name else _face_lv(wire.plane_name)
        plane_id = wire.plane_name or ""
        face_plane_wires.setdefault(face_id, {}).setdefault(plane_id, []).append(wire)

    faces = []
    for face_name, plane_dict in face_plane_wires.items():
        face = FaceGeom(name=face_name)
        for plane_name, wires in plane_dict.items():
            face.planes.append(PlaneGeom(name=plane_name, wires=list(wires)))
        faces.append(face)
    return faces


def sort_wires_by_pitch(plane_geom) -> list:
    """Return wires in a plane sorted by ascending pitch coordinate.

    The pitch direction is determined geometrically as the principal axis of
    wire-midpoint displacement perpendicular to the wire direction.

    Args:
        plane_geom: A :class:`PlaneGeom` whose wires all have valid
                    ``tail`` and ``head`` endpoints.

    Returns:
        New list of :class:`WireGeom` objects (same objects, reordered).
        An empty or single-wire plane is returned as-is.
    """
    wires = plane_geom.wires
    if len(wires) <= 1:
        return list(wires)

    # Determine wire direction from the first valid wire
    wire_dir = None
    for w in wires:
        d = w.head - w.tail
        n = np.linalg.norm(d)
        if n > 1e-9:
            wire_dir = d / n
            break
    if wire_dir is None:
        return list(wires)

    midpoints = np.array([0.5 * (w.head + w.tail) for w in wires])
    centroid = midpoints.mean(axis=0)
    centered = midpoints - centroid

    # Project out the wire-direction component to get the pitch displacement
    perp = centered - np.outer(centered @ wire_dir, wire_dir)

    # Principal axis of perp displacements = pitch direction (via SVD)
    _, _, vt = np.linalg.svd(perp, full_matrices=False)
    pitch_dir = vt[0]

    # Fix sign: ensure the largest-magnitude component is positive so that
    # the sort is consistent (ascending = toward more-positive dominant axis).
    dominant = np.argmax(np.abs(pitch_dir))
    if pitch_dir[dominant] < 0:
        pitch_dir = -pitch_dir

    projections = perp @ pitch_dir
    return [w for _, w in sorted(zip(projections, wires))]


def sort_planes_by_drift(face_geom) -> list:
    """Return planes sorted in U→V→W (first-induction to collection) order.

    The drift direction is inferred from the cross product of wire directions
    belonging to two non-parallel planes.  The cross product points away from
    the cathode (toward the collection plane).  Planes are sorted by
    *increasing* projection onto this direction so that U (closest to cathode)
    comes first.

    Args:
        face_geom: A :class:`FaceGeom` containing planes with wires.

    Returns:
        New list of :class:`PlaneGeom` objects (same objects, reordered).
        A single-plane face is returned as-is.
    """
    planes = face_geom.planes
    if len(planes) <= 1:
        return list(planes)

    def _wire_dir(plane):
        for w in plane.wires:
            d = w.head - w.tail
            n = np.linalg.norm(d)
            if n > 1e-9:
                return d / n
        return None

    dirs = [_wire_dir(p) for p in planes]

    # Find drift direction as cross product of first two non-parallel wire dirs
    drift_dir = None
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            if dirs[i] is not None and dirs[j] is not None:
                cross = np.cross(dirs[i], dirs[j])
                n = np.linalg.norm(cross)
                if n > 1e-9:
                    drift_dir = cross / n
                    break
        if drift_dir is not None:
            break

    if drift_dir is None:
        return list(planes)

    def _mean_pos(plane):
        mids = [0.5 * (w.head + w.tail) for w in plane.wires]
        return np.mean(mids, axis=0)

    projections = [_mean_pos(p) @ drift_dir for p in planes]
    return [p for _, p in sorted(zip(projections, planes))]


def _face_mean_pos(face) -> np.ndarray:
    """Return mean world-frame position of all wire midpoints in a face."""
    mids = [0.5 * (w.head + w.tail) for p in face.planes for w in p.wires]
    return np.mean(mids, axis=0) if mids else np.zeros(3)


def pair_faces_into_anodes(faces: list, connectivity_mode: str) -> list:
    """Group :class:`FaceGeom` objects into :class:`AnodeGeom` pairs.

    Args:
        faces:              List of :class:`FaceGeom` from
                            :func:`build_detector_faces`.
        connectivity_mode:  ``"vd"`` or ``"hd"``.

            * **VD** — *open-book*: pairs share the same TPC logical-volume
              name (encoded in ``face.name`` as ``"<lv>_<tag>"``).  The LV
              name is recovered by dropping the last ``_``-delimited token.
              Faces from the same LV are paired together.
            * **HD** — *closed-book*: pairs share the same Y-Z centroid.
              Faces are clustered by their rounded (Y, Z) mean position and
              paired within each cluster.

    Returns:
        List of :class:`AnodeGeom` objects.  Each anode contains exactly
        two faces.

    Raises:
        ValueError: If any cluster contains an odd number of faces (unpaired).
        ValueError: If *connectivity_mode* is not ``"vd"`` or ``"hd"``.
    """
    if not faces:
        return []

    if connectivity_mode == "vd":
        groups = _group_by_lv_name(faces)
    elif connectivity_mode == "hd":
        groups = _group_by_yz_centroid(faces)
    else:
        raise ValueError(
            f"Unknown connectivity_mode {connectivity_mode!r}. "
            "Must be 'vd' or 'hd'."
        )

    anodes = []
    for key, group in groups.items():
        if len(group) % 2 != 0:
            raise ValueError(
                f"Odd number of faces ({len(group)}) in group {key!r}; "
                "cannot form complete anode pairs."
            )
        for i in range(0, len(group), 2):
            anode = AnodeGeom(faces=list(group[i:i + 2]))
            anodes.append(anode)
    return anodes


def _group_by_lv_name(faces: list) -> dict:
    """Group faces by TPC logical-volume name (VD open-book pairing)."""
    groups: dict = {}
    for face in faces:
        # face.name is the physvol placement name, e.g. "volTPC0_top".
        # The LV name is everything before the last underscore-prefixed token.
        parts = face.name.rsplit("_", 1)
        lv_name = parts[0] if len(parts) == 2 else face.name
        groups.setdefault(lv_name, []).append(face)
    return groups


def _group_by_yz_centroid(faces: list, mm_tolerance: float = 1.0) -> dict:
    """Group faces by rounded Y-Z centroid (HD closed-book pairing)."""
    groups: dict = {}
    for face in faces:
        pos = _face_mean_pos(face)
        key = (round(pos[1] / mm_tolerance), round(pos[2] / mm_tolerance))
        groups.setdefault(key, []).append(face)
    return groups


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
