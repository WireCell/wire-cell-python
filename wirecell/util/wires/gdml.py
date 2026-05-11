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

from typing import Optional, Union

import numpy as np
from scipy.spatial.transform import Rotation

from .geom import *

def _gdml_float(s, default: float = 0.0, ns: dict = None) -> float:
    """Parse a GDML numeric attribute string, evaluating simple arithmetic.

    GDML files may use expressions like ``'0.5*0.02'`` or named constants
    (e.g. ``'EWProfileLength'``) in solid attributes.  Pass *ns* to supply
    a mapping of name → float for GDML ``<constant>`` values.
    Falls back to *default* when *s* is ``None`` or empty.
    """
    if s is None or s == "":
        return default
    try:
        return float(s)
    except (ValueError, TypeError):
        import math as _math
        ctx = dict(vars(_math))
        if ns:
            ctx.update(ns)
        return float(eval(str(s), {"__builtins__": {}}, ctx))


# ── Detector config ────────────────────────────────────────────────────────────

_REQUIRED_CONFIG_KEYS = frozenset({"role_patterns", "connectivity_mode", "nearness_tolerance"})

BUILTIN_CONFIGS: dict[str, dict] = {
    "protodunevd_v4": {
        "role_patterns": {
            "wire":     r"volTPCWire[UVZ]\d+",
            "plane":    r"volTPCPlane[UVZ]\d+",
            "face":     r"volTPC\d+",
            "detector": r"volCryostat",
        },
        "connectivity_mode": "vd",
        "nearness_tolerance": 0.1,
    },
    "protodunevd_v5": {
        "role_patterns": {
            # v5 U/V wire LVs have a mandatory _N suffix (volTPCWireU0_0)
            # while Z wire LVs do not (volTPCWireZ0).  The alternation makes
            # v4 U/V names (no suffix) fail to match.
            "wire":     r"volTPCWire([UV]\d+_\d+|Z\d+)",
            "plane":    r"volTPCPlane[UVZ]_\d+",
            "face":     r"volTPC_\d+",
            "detector": r"volCryostat",
        },
        "connectivity_mode": "vd",
        # v5 wire endpoints between anode faces are ~0.2 mm apart; 0.5 mm gives
        # comfortable margin while staying well below the ~4.75 mm wire pitch.
        "nearness_tolerance": 0.5,
    },
    "protodunehd_v8": {
        "role_patterns": {
            # U/V wires: numeric suffix (volTPCWireU0Inner) OR "Common" suffix
            # (volTPCWireUCommonInner) for the wrapping seg=2 wires.
            # Collection wires: volTPCWireVertInner/Outer.
            "wire":     r"volTPCWire([UV]\w+|Vert)(Inner|Outer)",
            "plane":    r"volTPCPlane[UVZ](Inner|Outer)",
            "face":     r"volTPC(Inner|Outer)",
            "detector": r"volCryostat",
        },
        "connectivity_mode": "hd",
        # HD uses cluster-range matching; nearness_tolerance is a tiny
        # floating-point buffer around the cluster edges (not a proximity radius).
        "nearness_tolerance": 0.1,
    },
}


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


def assign_vd_channels(anodes: list, connectivity: dict) -> dict:
    """Assign globally unique channel IDs to all wires in a VD detector.

    Channels are assigned sequentially across anodes.  Within each anode the
    iteration order is: face1 (directly connected to electronics, seg=0) then
    face0 (far face, seg=1), each face traversed plane-by-plane in
    :func:`sort_planes_by_drift` order and wire-by-wire in
    :func:`sort_wires_by_pitch` order.

    For each unassigned wire:

    * A new channel ID is allocated and assigned to the wire.
    * If the wire has a cross-face partner (``connected_to`` is not ``None``)
      and that partner has not yet been assigned, the **same** channel ID is
      given to the partner.

    Collection wires (``connected_to`` is ``None``) each receive a unique
    channel ID.

    Args:
        anodes:       List of :class:`AnodeGeom` in detector order.
        connectivity: Combined connectivity dict
                      ``{wire_name: {"segment": int, "connected_to": str|None}}``
                      as produced by :func:`find_vd_connected_pairs` (merged
                      over all anodes).

    Returns:
        ``dict[str, int]`` mapping every wire name to its channel ID.
    """
    result: dict = {}
    channel = 0

    for anode in anodes:
        # face1 (index 1) carries seg=0 for induction wires → process first so
        # the primary assignment lands on the direct-electronics face.
        for face in reversed(anode.faces):
            for plane in sort_planes_by_drift(face):
                for wire in sort_wires_by_pitch(plane):
                    if wire.name in result:
                        continue
                    result[wire.name] = channel
                    partner = connectivity.get(wire.name, {}).get("connected_to")
                    if partner is not None and partner not in result:
                        result[partner] = channel
                    channel += 1

    return result


# ── HD connectivity ────────────────────────────────────────────────────────────

def _find_z_boundaries(anode_geom, bin_size: float = 1.0, density_factor: float = 10.0) -> list:
    """Find Z boundary regions where wire endpoints cluster far more densely than average.

    Wire wrapping in HD geometry creates large spikes in the Z-endpoint
    distribution at a handful of boundary locations.  This function detects
    those spikes by histogramming all endpoint Z values and returning
    ``(z_min, z_max)`` tuples — the actual min/max Z of endpoints in each
    high-density cluster.

    Returns:
        Sorted list of ``(z_min, z_max)`` tuples (in mm).  Use these as
        acceptance windows with a small extra buffer for floating-point margin.
    """
    z_vals = []
    for face in anode_geom.faces:
        for plane in face.planes:
            for wire in plane.wires:
                for pt in (wire.tail, wire.head):
                    if pt is not None:
                        z_vals.append(pt[2])
    if not z_vals:
        return []

    z_arr = np.array(z_vals)
    z_min, z_max = float(z_arr.min()), float(z_arr.max())
    edges = np.arange(z_min - bin_size, z_max + 2 * bin_size, bin_size)
    counts, edges = np.histogram(z_arr, bins=edges)

    mean_count = float(counts.mean())
    if mean_count < 1:
        return []

    hot = np.where(counts >= density_factor * mean_count)[0]
    if len(hot) == 0:
        return []

    # Cluster adjacent hot bins and record the actual Z range of each cluster.
    boundaries = []
    i = 0
    while i < len(hot):
        j = i
        while j + 1 < len(hot) and hot[j + 1] - hot[j] <= 2:
            j += 1
        cluster = hot[i:j + 1]
        z_lo = edges[cluster[0]]
        z_hi = edges[cluster[-1] + 1]
        cluster_pts = z_arr[(z_arr >= z_lo) & (z_arr <= z_hi)]
        if len(cluster_pts) > 0:
            boundaries.append((float(cluster_pts.min()), float(cluster_pts.max())))
        i = j + 1

    return sorted(boundaries)


def _hd_wire_segment(wire, z_boundaries: list, z_buf: float = 0.1) -> int:
    """Return the segment value (0, 1, or 2) for an HD wire from endpoint geometry.

    * **0** — head is NOT within any Z boundary region (primary wire).
    * **1** — BOTH tail and head fall within Z boundary regions (jumper wire).
    * **2** — head IS within a Z boundary region, tail is NOT.

    Args:
        z_boundaries: List of ``(z_min, z_max)`` tuples from
                      :func:`_find_z_boundaries`.
        z_buf:        Extra buffer in mm added to each boundary's z_min/z_max
                      to absorb floating-point rounding (default 0.1 mm).
    """
    def _near(pt) -> bool:
        if pt is None:
            return False
        return any((z_lo - z_buf) <= pt[2] <= (z_hi + z_buf)
                   for z_lo, z_hi in z_boundaries)

    head_near = _near(wire.head)
    tail_near = _near(wire.tail)
    if head_near and tail_near:
        return 1
    if head_near:
        return 2
    return 0


def _find_hd_z_pairs(anode_geom, z_boundaries: list, z_buf: float = 0.1) -> list:
    """Return (face0_wire_name, face1_wire_name) pairs connected at Z boundaries.

    At each Z boundary region, wire endpoints from face0 and face1 whose Z
    coordinate falls within ``[z_min - z_buf, z_max + z_buf]`` are sorted by Y
    descending and matched by rank.  Collection planes are skipped.  Each wire
    contributes at most one endpoint per boundary region.

    Args:
        z_boundaries: List of ``(z_min, z_max)`` tuples from
                      :func:`_find_z_boundaries`.
        z_buf:        Extra buffer in mm added around each boundary's z_min/z_max
                      to absorb floating-point rounding (default 0.1 mm).

    ``anode_geom.faces[0]`` is face0 (higher X), ``anode_geom.faces[1]`` is
    face1 (lower X), matching the convention set by :func:`pair_faces_into_anodes`.
    """
    pairs: list = []
    face0, face1 = anode_geom.faces[0], anode_geom.faces[1]

    for (z_lo, z_hi) in z_boundaries:
        face0_pts: list = []  # (y_value, wire_name)
        face1_pts: list = []

        for face_idx, face in enumerate((face0, face1)):
            pts_list = face0_pts if face_idx == 0 else face1_pts
            seen: set = set()
            for plane in face.planes:
                if _is_collection_plane(plane):
                    continue
                for wire in plane.wires:
                    if wire.name in seen:
                        continue
                    for pt in (wire.tail, wire.head):
                        if pt is not None and (z_lo - z_buf) <= pt[2] <= (z_hi + z_buf):
                            pts_list.append((pt[1], wire.name))
                            seen.add(wire.name)
                            break  # at most one endpoint per wire per boundary

        face0_pts.sort(key=lambda x: -x[0])  # descending Y
        face1_pts.sort(key=lambda x: -x[0])  # descending Y

        for (_, n0), (_, n1) in zip(face0_pts, face1_pts):
            pairs.append((n0, n1))

    return pairs


def _build_connected_components(wire_names: list, pairs: list) -> dict:
    """Build connected components from a list of (name_a, name_b) edge pairs.

    Returns:
        ``dict[str, frozenset]`` — maps each wire name to the frozenset of
        all wire names in the same connected component (including itself).
    """
    parent = {name: name for name in wire_names}

    def _find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x: str, y: str) -> None:
        px, py = _find(x), _find(y)
        if px != py:
            parent[px] = py

    for n0, n1 in pairs:
        if n0 in parent and n1 in parent:
            _union(n0, n1)

    groups: dict = {}
    for name in wire_names:
        root = _find(name)
        groups.setdefault(root, set()).add(name)

    result: dict = {}
    for members in groups.values():
        fs = frozenset(members)
        for name in members:
            result[name] = fs
    return result


def _assign_hd_segments_topological(
    all_wire_names: list,
    pairs: list,
    wire_seg_geometric: dict,
) -> dict:
    """Return topologically-corrected segment values for HD wires.

    The purely geometric rule in :func:`_hd_wire_segment` misclassifies wires
    in 2-wire connected components: a "connector" wire whose head is at one Z
    boundary and whose tail is interior gets geometric seg=2, but is only one
    hop from the electronics-facing root wire and should be seg=1.  Standalone
    wires whose head happens to land near a Z boundary are similarly
    misclassified as seg=2 instead of seg=0.

    This function corrects both cases by doing a BFS from the geometric-seg=0
    root(s) within each connected component and assigning the BFS distance as
    the segment value.

    Rules:
    * Component of size 1 → seg=0 (standalone; always directly on electronics).
    * Root wire(s) (geometric seg=0) within larger components → seg=0.
    * Neighbors of roots → seg=1.
    * Neighbors of their neighbors → seg=2.

    Args:
        all_wire_names:    All wire names in the anode (including isolated ones).
        pairs:             ``(name_a, name_b)`` edges from Z-boundary matching.
        wire_seg_geometric: ``{wire_name: geometric_segment}`` from
                            :func:`_hd_wire_segment`.

    Returns:
        ``dict[str, int]`` — corrected segment for every wire name.
    """
    # Build adjacency from pairs.
    adjacency: dict = {name: set() for name in all_wire_names}
    for n0, n1 in pairs:
        if n0 in adjacency and n1 in adjacency:
            adjacency[n0].add(n1)
            adjacency[n1].add(n0)

    component_of = _build_connected_components(all_wire_names, pairs)

    # Deduplicate components (component_of maps every wire to its frozenset).
    seen: set = set()
    components: list = []
    for fs in component_of.values():
        if fs not in seen:
            seen.add(fs)
            components.append(fs)

    result: dict = {}
    for fs in components:
        component = list(fs)

        if len(component) == 1:
            result[component[0]] = 0
            continue

        roots = [w for w in component if wire_seg_geometric.get(w, -1) == 0]
        if not roots:
            # Orphaned component: no geo_seg=0 root because the natural seg=0
            # partner was left unmatched by Y-rank pairing (count mismatch at a
            # Z boundary).  The geo_seg=2 wire (one boundary endpoint, one
            # interior endpoint) is the electronics-facing primary; treat it as
            # the BFS root (seg=0) and assign its geo_seg=1 neighbour seg=1.
            alt_roots = [w for w in component if wire_seg_geometric.get(w, -1) == 2]
            if alt_roots:
                alt_dist: dict = {w: 0 for w in alt_roots}
                alt_queue = list(alt_roots)
                alt_head = 0
                while alt_head < len(alt_queue):
                    cur = alt_queue[alt_head]
                    alt_head += 1
                    for nb in adjacency[cur]:
                        if nb not in alt_dist:
                            alt_dist[nb] = alt_dist[cur] + 1
                            alt_queue.append(nb)
                for name in component:
                    result[name] = alt_dist.get(name, wire_seg_geometric.get(name, 0))
                continue
            for name in component:
                result[name] = wire_seg_geometric.get(name, 0)
            continue

        # BFS from all roots simultaneously.
        dist: dict = {w: 0 for w in roots}
        queue = list(roots)
        head_idx = 0
        while head_idx < len(queue):
            current = queue[head_idx]
            head_idx += 1
            for neighbor in adjacency[current]:
                if neighbor not in dist:
                    dist[neighbor] = dist[current] + 1
                    queue.append(neighbor)

        for name in component:
            result[name] = dist.get(name, wire_seg_geometric.get(name, 0))

    return result


def _orient_hd_wires(anodes: list) -> None:
    """Set head/tail of every WireGeom in HD anodes to the physical convention.

    HD convention: head = the endpoint with larger Y coordinate (toward
    electronics readout at the top).  ``extract_wires`` uses a geometric
    default (head = local +Z axis endpoint) which is arbitrary and breaks the
    Z-boundary segment classification that ``build_hd_channel_map`` relies on.
    This function fixes the orientation in-place before channel assignment.
    """
    for anode in anodes:
        for face in anode.faces:
            for plane in face.planes:
                for wire in plane.wires:
                    if wire.tail[1] > wire.head[1]:
                        wire.head, wire.tail = wire.tail, wire.head


def build_hd_channel_map(anodes: list, z_buf: float = 0.1) -> dict:
    """Assign channel IDs and segment values to all wires in HD anodes.

    Channel allocation strategy (must reproduce the reference file convention):

    * Z boundary *regions* (z_min, z_max) are found from the endpoint distribution.
    * Segment (0/1/2) is determined geometrically per wire using those regions.
    * Connected components are built via Z-boundary rank matching.
    * Channels are allocated per anode: face1 (lower-X, index 1) first then
      face0 (higher-X, index 0), each traversed in
      :func:`sort_planes_by_drift` × :func:`sort_wires_by_pitch` order.
    * When a wire's component has not yet been fully assigned, the new channel
      is propagated to all un-assigned component members.

    Args:
        anodes:  List of :class:`AnodeGeom`, with ``faces[0]`` = higher-X face
                 and ``faces[1]`` = lower-X face (as set by
                 :func:`pair_faces_into_anodes`).
        z_buf:   Extra buffer in mm added around each boundary region when
                 checking endpoint proximity (absorbs floating-point noise).

    Returns:
        ``dict[str, (int, int)]`` — ``{wire_name: (channel_id, segment)}``.
    """
    channel_map: dict = {}
    channel = 0

    for anode in anodes:
        z_bounds = _find_z_boundaries(anode)

        # Collect all wire names and compute geometric segments.
        all_wire_names: list = []
        wire_seg_geo: dict = {}
        for face in anode.faces:
            for plane in face.planes:
                for wire in plane.wires:
                    all_wire_names.append(wire.name)
                    wire_seg_geo[wire.name] = _hd_wire_segment(wire, z_bounds, z_buf)

        pairs = _find_hd_z_pairs(anode, z_bounds, z_buf)
        # Topological correction: 2-wire-component connectors get seg=1 not seg=2.
        wire_seg = _assign_hd_segments_topological(all_wire_names, pairs, wire_seg_geo)
        component_of = _build_connected_components(all_wire_names, pairs)

        assigned: set = set()
        # Iterate plane-by-plane (U, V, W) and within each plane process
        # face1 (lower-X, faces[1]) before face0 (higher-X, faces[0]).
        #
        # Pitch direction convention — the reference channel assignment uses:
        #   U (plane_idx=0): face1 DESCENDING, face0 ASCENDING
        #   V (plane_idx=1): face1 ASCENDING,  face0 DESCENDING
        #   W (plane_idx≥2): both ASCENDING
        # The alternating U/V pattern arises because the inner-TPC rotation
        # (180° about X+Y) swaps the local U and V wire orientations so that
        # inner-TPC V wires appear as outer-TPC U wires in the world frame.
        #
        # A new channel is allocated only when the seg=0 (primary) wire of a
        # component is encountered; connectors (seg=1) and trailing ends
        # (seg=2) are assigned at that same time.
        face0 = anode.faces[0]  # higher-X (Inner TPC)
        face1 = anode.faces[1]  # lower-X (Outer TPC)
        planes0 = sort_planes_by_drift(face0)
        planes1 = sort_planes_by_drift(face1)
        for plane_idx, (plane1, plane0) in enumerate(zip(planes1, planes0)):
            for inner_face_idx, (face, plane) in enumerate([(face1, plane1), (face0, plane0)]):
                # Descending pitch for U-face1 and V-face0 (induction planes only).
                descending = (plane_idx < 2) and (plane_idx % 2 == inner_face_idx % 2)
                wires_ordered = sort_wires_by_pitch(plane)
                if descending:
                    wires_ordered = list(reversed(wires_ordered))
                for wire in wires_ordered:
                    if wire.name in assigned:
                        continue
                    if wire_seg[wire.name] != 0:
                        continue
                    component = component_of[wire.name]
                    for partner in component:
                        if partner not in assigned:
                            channel_map[partner] = (channel, wire_seg[partner])
                            assigned.add(partner)
                    channel += 1

    return channel_map


def build_store(anodes: list, channel_map: dict):
    """Build a :class:`~wirecell.util.wires.schema.Store` from intermediate geometry.

    Args:
        anodes:      List of :class:`AnodeGeom` in detector order, as returned
                     by :func:`pair_faces_into_anodes`.
        channel_map: Mapping ``{wire_name: (channel_id, segment)}`` combining the
                     results of :func:`assign_vd_channels` (for channel IDs) and
                     :func:`find_vd_connected_pairs` (for segment values).

    Returns:
        A :class:`wirecell.util.wires.schema.Store` with one
        :class:`~wirecell.util.wires.schema.Detector` containing all anodes,
        faces, planes, wires, and points.

    Hierarchy:
        * Anode ``ident`` = global anode index (0-based).
        * Face ``ident`` = face position within the anode (0 or 1).
        * Plane ``ident`` = 0/1/2 for U/V/W (drift order).
        * Wire ``ident`` = wire index in pitch-sorted order within its plane.
        * Points are deduplicated at 0.001 mm precision (µm-level).
    """
    from wirecell.util.wires import schema as wschema

    pts: list = []
    wires: list = []
    planes: list = []
    faces: list = []
    s_anodes: list = []

    _pt_key: dict = {}

    def _point(xyz) -> int:
        key = (round(float(xyz[0]), 3), round(float(xyz[1]), 3), round(float(xyz[2]), 3))
        if key not in _pt_key:
            _pt_key[key] = len(pts)
            pts.append(wschema.Point(x=key[0], y=key[1], z=key[2]))
        return _pt_key[key]

    for anode_idx, anode in enumerate(anodes):
        face_indices: list = []

        for face_idx, face in enumerate(anode.faces):
            plane_indices: list = []

            for plane_ident, plane in enumerate(sort_planes_by_drift(face)):
                wire_indices: list = []

                for wire_ident, geom_wire in enumerate(sort_wires_by_pitch(plane)):
                    channel_id, segment = channel_map.get(geom_wire.name, (0, 0))
                    tail_idx = _point(geom_wire.tail)
                    head_idx = _point(geom_wire.head)
                    wire_store_idx = len(wires)
                    wires.append(wschema.Wire(
                        ident=wire_ident,
                        channel=channel_id,
                        segment=segment,
                        tail=tail_idx,
                        head=head_idx,
                    ))
                    wire_indices.append(wire_store_idx)

                plane_store_idx = len(planes)
                planes.append(wschema.Plane(ident=plane_ident, wires=wire_indices))
                plane_indices.append(plane_store_idx)

            face_store_idx = len(faces)
            faces.append(wschema.Face(ident=face_idx, planes=plane_indices))
            face_indices.append(face_store_idx)

        s_anodes.append(wschema.Anode(ident=anode_idx, faces=face_indices))

    detector = wschema.Detector(ident=0, anodes=list(range(len(s_anodes))))
    return wschema.Store(
        detectors=[detector],
        anodes=s_anodes,
        faces=faces,
        planes=planes,
        wires=wires,
        points=pts,
    )


def convert(gdml_path, config: dict, root_vol: str = ""):
    """Orchestrate the full GDML → wires-schema pipeline.

    Args:
        gdml_path: Path to the GDML file (``str`` or :class:`pathlib.Path`).
        config:    Detector config dict as returned by :func:`load_config`.
        root_vol:  Starting logical volume name.  Defaults to the world volume
                   declared in the GDML ``<setup>`` section.

    Returns:
        A :class:`~wirecell.util.wires.schema.Store` for the full detector.

    Raises:
        ValueError: If no root volume can be determined.
        NotImplementedError: If ``config["connectivity_mode"]`` is not ``"vd"``.
    """
    import xml.etree.ElementTree as ET
    from xml.etree.ElementTree import ParseError
    try:
        gdml_root = ET.parse(str(gdml_path)).getroot()
    except ParseError:
        print(f'error reading {gdml_path}')
        raise
    defines = parse_define(gdml_root)
    solids = parse_solids(gdml_root, defines)
    vol_tree = parse_structure(gdml_root, defines, solids)

    if not root_vol:
        setup = gdml_root.find("setup")
        if setup is not None:
            world_el = setup.find("world")
            if world_el is not None:
                root_vol = world_el.get("ref", "")
    if not root_vol:
        raise ValueError(
            "Could not determine root volume from GDML <setup>; "
            "pass root_vol explicitly."
        )

    patterns = config["role_patterns"]
    wires = extract_wires(vol_tree, defines, solids, root_vol, patterns)
    faces = build_detector_faces(vol_tree, wires, patterns)
    anodes = pair_faces_into_anodes(faces, config["connectivity_mode"])

    tolerance = float(config["nearness_tolerance"])
    mode = config["connectivity_mode"]

    if mode == "vd":
        connectivity: dict = {}
        for anode in anodes:
            connectivity.update(find_vd_connected_pairs(anode, tolerance))
        channels = assign_vd_channels(anodes, connectivity)
        channel_map = {
            name: (channels[name], connectivity.get(name, {}).get("segment", 0))
            for name in channels
        }
    elif mode == "hd":
        _orient_hd_wires(anodes)
        channel_map = build_hd_channel_map(anodes, z_buf=tolerance)
    else:
        raise NotImplementedError(
            f"connectivity_mode={mode!r} is not yet implemented."
        )

    return build_store(anodes, channel_map)


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
        return {"positions": positions, "rotations": rotations, "constants": {}}

    # First pass: collect <constant> values (may be referenced in positions/rotations)
    constants: dict[str, float] = {}
    for elem in define:
        if elem.tag == "constant":
            cname = elem.get("name")
            val = elem.get("value")
            if cname and val is not None:
                try:
                    constants[cname] = float(val)
                except (ValueError, TypeError):
                    pass

    for elem in define:
        name = elem.get("name")
        if name is None:
            continue
        if elem.tag == "position":
            unit = elem.get("unit", "mm")
            scale = _UNIT_TO_MM.get(unit, 1.0)
            x = _gdml_float(elem.get("x"), ns=constants) * scale
            y = _gdml_float(elem.get("y"), ns=constants) * scale
            z = _gdml_float(elem.get("z"), ns=constants) * scale
            positions[name] = np.array([x, y, z], dtype=float)
        elif elem.tag == "rotation":
            unit = elem.get("unit", "deg")
            scale = _UNIT_TO_RAD.get(unit, np.pi / 180.0)
            rx = _gdml_float(elem.get("x"), ns=constants) * scale
            ry = _gdml_float(elem.get("y"), ns=constants) * scale
            rz = _gdml_float(elem.get("z"), ns=constants) * scale
            rotations[name] = np.array([rx, ry, rz], dtype=float)

    return {"positions": positions, "rotations": rotations, "constants": constants}


def parse_solids(gdml_root, defines: dict = None) -> dict:
    """Parse the GDML <solids> section and return tube dimensions.

    Only ``<tube>`` elements are returned; box and other solid types are
    ignored.

    Args:
        gdml_root: The root Element of a parsed GDML document.
        defines:   Optional output of :func:`parse_define`.  When provided its
                   ``"constants"`` entry is used to resolve named values such as
                   ``z="EWProfileLength"`` that appear in some GDML files.

    Returns:
        ``dict[str, dict]`` mapping each tube solid name to::

            {"rmax": float,   # outer radius in mm
             "half_z": float} # half-length in mm (GDML z attribute / 2)
    """
    _UNIT_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0}

    constants = (defines or {}).get("constants", {})
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
        rmax = _gdml_float(elem.get("rmax"), ns=constants) * scale
        full_z = _gdml_float(elem.get("z"), ns=constants) * scale
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
            [_gdml_float(elem.get("x")) * scale,
             _gdml_float(elem.get("y")) * scale,
             _gdml_float(elem.get("z")) * scale],
            dtype=float,
        )

    def _rot_from_elem(elem) -> np.ndarray:
        unit = elem.get("unit", "deg")
        scale = _UNIT_TO_RAD.get(unit, np.pi / 180.0)
        return np.array(
            [_gdml_float(elem.get("x")) * scale,
             _gdml_float(elem.get("y")) * scale,
             _gdml_float(elem.get("z")) * scale],
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

    # Counter for anonymous face placements (physvol name == LV name).
    _face_counter: dict = {}
    # Counter for repeated wire LV placements within the same face.
    # Key: (face_name, vol_name) → number of times this LV has been placed.
    # When a single wire LV (e.g. volTPCWireZ0) is placed N times in one face,
    # the second and subsequent placements get a numeric suffix to stay unique.
    _wire_counter: dict = {}

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
                # Build a unique wire name.  Prepend the face placement name so
                # the same wire LV in multiple TPC instances stays distinct.
                # When the same wire LV is placed >1 times in the same face
                # (e.g. a single volTPCWireZ0 placed 292 times in one Z plane),
                # append a 1-based counter to the 2nd and later occurrences so
                # that all wire names in the detector are unique.
                base = f"{face_name}.{vol_name}" if face_name else vol_name
                key = (face_name, vol_name)
                n = _wire_counter.get(key, 0)
                _wire_counter[key] = n + 1
                wire_name = base if n == 0 else f"{base}_{n}"
                result.append(WireGeom(
                    name=wire_name,
                    tail=tail,
                    head=head,
                    radius=dims["rmax"],
                    plane_name=plane_name,
                    face_name=face_name,
                ))
            return  # leaf — do not recurse into wire volumes

        if role == "face":
            if pv_name != vol_name:
                # Explicit physvol name (e.g. "volTPC0_top") — use as-is.
                face_name = pv_name
            else:
                # Anonymous physvol (same name as LV) — append a counter to
                # make each placement unique (e.g. "volTPC0_0", "volTPC0_1").
                n = _face_counter.get(vol_name, 0)
                _face_counter[vol_name] = n + 1
                face_name = f"{vol_name}_{n}"
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

            * **VD** — *open-book*: faces that share the same X and Z centroid
              (within 100 mm tolerance) belong to the same anode column.  Within
              each column, faces are sorted by Y centroid and paired adjacent
              (faces 0+1, faces 2+3, …) so that the two nearest Y positions form
              one anode.  This matches the ProtoDUNE-VD vertical-drift geometry
              where two faces sit side-by-side in Y at the same (X, Z) location
              with the 180° rotation about Y creating the open-book orientation.
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
        # Group by (X, Z) column, then pair adjacent faces by ascending Y.
        # Within each pair: face[0] = higher-Y face, face[1] = lower-Y face,
        # matching the reference convention where face0 is the "inner" face
        # (closer to the other anode in the Y direction).
        xz_groups = _group_by_xz_centroid(faces)
        groups: dict = {}
        for xz_key, xz_faces in xz_groups.items():
            sorted_faces = sorted(xz_faces, key=lambda f: _face_mean_pos(f)[1])
            for pair_idx in range(0, len(sorted_faces), 2):
                pair = sorted_faces[pair_idx:pair_idx + 2]
                if len(pair) == 2:
                    # Reverse so higher-Y face is first.
                    groups[(xz_key, pair_idx // 2)] = [pair[1], pair[0]]
                else:
                    # Odd-count group — store as-is; the error check below will catch it.
                    groups[(xz_key, pair_idx // 2)] = pair
    elif connectivity_mode == "hd":
        # HD closed-book: pair Inner + Outer TPC faces that share the same
        # X-side and Z-position.  x_tol=200 mm collapses Inner/Outer (~100 mm
        # apart) into one key while separating left/right sides (>>1 m apart).
        # z_tol=100 mm collapses the ~0.5 mm Inner/Outer Z offset while keeping
        # distinct anode rows (hundreds of mm apart) separate.
        xz_raw = _group_by_xz_centroid(faces, x_tol=200.0, z_tol=100.0)
        groups = {}
        for key, group_faces in xz_raw.items():
            # Sort descending X: face[0] = higher-X face (ident 0),
            # face[1] = lower-X face (ident 1), matching reference convention.
            groups[key] = sorted(group_faces, key=lambda f: _face_mean_pos(f)[0], reverse=True)
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

    # Sort anodes by their mean world position so the output order is
    # deterministic and matches the reference file ordering.
    # Round to the nearest 100 mm to suppress floating-point noise.
    # VD: sort by (X, Y, Z) — drift columns first.
    # HD: sort by (Z, X, Y) — beam-direction rows first, then drift side.
    def _anode_sort_key(a):
        pos = _anode_mean_pos(a)
        x, y, z = (round(float(v), -2) for v in pos)
        if connectivity_mode == "hd":
            return (z, x, y)
        return (x, y, z)

    anodes.sort(key=_anode_sort_key)
    return anodes


def _anode_mean_pos(anode) -> np.ndarray:
    """Return mean world-frame position of all wires in an anode."""
    poses = [_face_mean_pos(f) for f in anode.faces]
    return np.mean(poses, axis=0)


def _group_by_xz_centroid(faces: list, x_tol: float = 100.0, z_tol: float = 100.0) -> dict:
    """Group faces by rounded X-Z centroid (VD open-book column grouping).

    VD open-book faces in the same anode column share the same X (drift side)
    and Z (beam position) coordinates.  The Y coordinate distinguishes the two
    faces within a pair (they are side-by-side in the drift-perpendicular Y
    direction).
    """
    groups: dict = {}
    for face in faces:
        pos = _face_mean_pos(face)
        key = (round(pos[0] / x_tol), round(pos[2] / z_tol))
        groups.setdefault(key, []).append(face)
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
