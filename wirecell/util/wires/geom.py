from dataclasses import dataclass, field
from typing import Optional

import numpy as np


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
    wires: list = field(default_factory=list)


@dataclass
class FaceGeom:
    """Intermediate representation of one TPC face (collection of planes)."""
    name: str
    planes: list = field(default_factory=list)


@dataclass
class AnodeGeom:
    """Intermediate representation of an anode (one or two faces)."""
    faces: list = field(default_factory=list)


@dataclass
class DetectorGeom:
    """Intermediate representation of a full detector (collection of anodes)."""
    anodes: list = field(default_factory=list)


# ── Geometry ordering helpers ──────────────────────────────────────────────────

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

    perp = centered - np.outer(centered @ wire_dir, wire_dir)

    _, _, vt = np.linalg.svd(perp, full_matrices=False)
    pitch_dir = vt[0]

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


# ── Schema ↔ DetectorGeom conversions ─────────────────────────────────────────

def togeom(store) -> DetectorGeom:
    """Reconstruct a :class:`DetectorGeom` from a :class:`~schema.Store`.

    Wire names are synthesised from the hierarchy idents
    (``"a{ai}.f{fi}.p{pi}.w{wi}"``).  Wire radius is set to ``0.0`` because
    the wires schema does not store it.  All other fields — channel, segment,
    and world-frame endpoints — are taken directly from the Store.

    Args:
        store: A :class:`~wirecell.util.wires.schema.Store`.

    Returns:
        A :class:`DetectorGeom` whose :class:`WireGeom` objects have
        ``channel`` and ``segment`` populated and ``tail``/``head`` as
        ``numpy`` arrays in mm.
    """
    det_geom = DetectorGeom()

    for det in store.detectors:
        for ai in det.anodes:
            anode = store.anodes[ai]
            anode_geom = AnodeGeom()

            for fi in anode.faces:
                face = store.faces[fi]
                face_name = f"a{anode.ident}.f{face.ident}"
                face_geom = FaceGeom(name=face_name)

                for pi in face.planes:
                    plane = store.planes[pi]
                    plane_name = f"{face_name}.p{plane.ident}"
                    plane_geom = PlaneGeom(name=plane_name)

                    for wi in plane.wires:
                        wire = store.wires[wi]
                        tp = store.points[wire.tail]
                        hp = store.points[wire.head]
                        wire_geom = WireGeom(
                            name=f"{plane_name}.w{wire.ident}",
                            tail=np.array([tp.x, tp.y, tp.z], dtype=float),
                            head=np.array([hp.x, hp.y, hp.z], dtype=float),
                            radius=0.0,
                            plane_name=plane_name,
                            face_name=face_name,
                            channel=wire.channel,
                            segment=wire.segment,
                        )
                        plane_geom.wires.append(wire_geom)

                    face_geom.planes.append(plane_geom)

                anode_geom.faces.append(face_geom)

            det_geom.anodes.append(anode_geom)

    return det_geom


def tostore(det_geom) -> 'schema.Store':
    """Build a :class:`~schema.Store` from a :class:`DetectorGeom`.

    Plane idents (0=U, 1=V, 2=W) are assigned by
    :func:`sort_planes_by_drift`.  Wire idents are assigned in
    :func:`sort_wires_by_pitch` order.  Channel and segment are taken from
    ``WireGeom.channel`` and ``WireGeom.segment`` (defaulting to 0 when
    ``None``).  Points are deduplicated at 0.001 mm precision.

    Args:
        det_geom: A :class:`DetectorGeom` whose wires have world-frame
                  ``tail`` and ``head`` endpoints.

    Returns:
        A :class:`~wirecell.util.wires.schema.Store` with one
        :class:`~wirecell.util.wires.schema.Detector`.
    """
    from . import schema as wschema

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

    for anode_idx, anode in enumerate(det_geom.anodes):
        face_indices: list = []

        for face_idx, face in enumerate(anode.faces):
            plane_indices: list = []

            for plane_ident, plane in enumerate(sort_planes_by_drift(face)):
                wire_indices: list = []

                for wire_ident, geom_wire in enumerate(sort_wires_by_pitch(plane)):
                    channel_id = geom_wire.channel if geom_wire.channel is not None else 0
                    segment = geom_wire.segment if geom_wire.segment is not None else 0
                    wire_store_idx = len(wires)
                    wires.append(wschema.Wire(
                        ident=wire_ident,
                        channel=channel_id,
                        segment=segment,
                        tail=_point(geom_wire.tail),
                        head=_point(geom_wire.head),
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
