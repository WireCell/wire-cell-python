"""VTK I/O for wire geometry.

Writes a :class:`~.geom.DetectorGeom` to either a flat VTP file or a nested
multiblock VTM file whose hierarchy matches the requested :class:`Blocking`
level.  The raw ``vtk`` Python bindings are used directly.

Typical usage::

    from wirecell.util.wires.vtkio import save, Blocking
    from wirecell.util.wires.geom import togeom

    det = togeom(store)
    save("mywires", det, Blocking.FACE)   # → mywires.vtm + mywires/

Public building-block functions (:func:`plane_polydata`, :func:`face_polydata`,
:func:`anode_polydata`, :func:`detector_polydata`, :func:`build_multiblock`,
:func:`write_vtp`, :func:`write_vtm`) are exposed for callers that need finer
control.
"""

from __future__ import annotations

import enum
import pathlib

import numpy as np

from .geom import (
    AnodeGeom,
    DetectorGeom,
    FaceGeom,
    PlaneGeom,
    WireGeom,
    sort_planes_by_drift,
    sort_wires_by_pitch,
)


# ── Blocking enum ──────────────────────────────────────────────────────────────

class Blocking(str, enum.Enum):
    """Controls the VTK block hierarchy written by :func:`save`.

    ``str`` mixin allows ``Blocking("anode")`` and ``"anode" == Blocking.ANODE``.
    """
    DETECTOR = "detector"
    ANODE    = "anode"
    FACE     = "face"
    PLANE    = "plane"


# ── Low-level polydata builder ─────────────────────────────────────────────────

def _build_polydata(wire_tuples) -> object:
    """Build a ``vtkPolyData`` from an iterable of wire tuples.

    Each tuple has the form ``(wire, anode_id, face_id, plane_id, wip)``
    where *wip* is the wire's 0-based index in ascending pitch order within
    its plane.  Wires with ``None`` endpoints are silently skipped.

    The returned ``vtkPolyData`` contains one ``vtkLine`` cell per wire and
    the following per-cell data arrays:

    * integer scalars: ``anode_id``, ``face_id``, ``plane_id``, ``wire_id``,
      ``channel_id``, ``segment``, ``wip``
    * double 3-vectors: ``tail``, ``head``, ``arrow``

    ``arrow`` is registered as the active vector attribute.
    """
    import vtk

    points = vtk.vtkPoints()
    points.SetDataTypeToDouble()
    lines = vtk.vtkCellArray()

    int_names = ("anode_id", "face_id", "plane_id", "wire_id",
                 "channel_id", "segment", "wip")
    int_arrs = {}
    for name in int_names:
        a = vtk.vtkIntArray()
        a.SetName(name)
        a.SetNumberOfComponents(1)
        int_arrs[name] = a

    vec_names = ("tail", "head", "arrow")
    vec_arrs = {}
    for name in vec_names:
        a = vtk.vtkDoubleArray()
        a.SetName(name)
        a.SetNumberOfComponents(3)
        vec_arrs[name] = a

    for wire, anode_id, face_id, plane_id, wip in wire_tuples:
        if wire.tail is None or wire.head is None:
            continue
        tail  = np.asarray(wire.tail,  dtype=float)
        head  = np.asarray(wire.head,  dtype=float)
        arrow = head - tail

        tid = points.InsertNextPoint(tail[0],  tail[1],  tail[2])
        hid = points.InsertNextPoint(head[0],  head[1],  head[2])

        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, tid)
        line.GetPointIds().SetId(1, hid)
        lines.InsertNextCell(line)

        int_arrs["anode_id"].InsertNextValue(anode_id)
        int_arrs["face_id"].InsertNextValue(face_id)
        int_arrs["plane_id"].InsertNextValue(plane_id)
        int_arrs["wire_id"].InsertNextValue(wip)
        int_arrs["channel_id"].InsertNextValue(
            wire.channel if wire.channel is not None else -1)
        int_arrs["segment"].InsertNextValue(
            wire.segment if wire.segment is not None else 0)
        int_arrs["wip"].InsertNextValue(wip)

        vec_arrs["tail"].InsertNextTuple3( tail[0],  tail[1],  tail[2])
        vec_arrs["head"].InsertNextTuple3( head[0],  head[1],  head[2])
        vec_arrs["arrow"].InsertNextTuple3(arrow[0], arrow[1], arrow[2])

    pd = vtk.vtkPolyData()
    pd.SetPoints(points)
    pd.SetLines(lines)

    cell_data = pd.GetCellData()
    for arr in int_arrs.values():
        cell_data.AddArray(arr)
    for arr in vec_arrs.values():
        cell_data.AddArray(arr)
    cell_data.SetActiveVectors("arrow")

    return pd


# ── Wire tuple generators ──────────────────────────────────────────────────────

def _iter_plane(plane: PlaneGeom, anode_id: int, face_id: int, plane_id: int):
    """Yield ``(wire, anode_id, face_id, plane_id, wip)`` for each wire in *plane*."""
    for wip, wire in enumerate(sort_wires_by_pitch(plane)):
        yield wire, anode_id, face_id, plane_id, wip


def _iter_face(face: FaceGeom, anode_id: int, face_id: int):
    """Yield wire tuples for every wire in *face*, planes in drift order."""
    for plane_id, plane in enumerate(sort_planes_by_drift(face)):
        yield from _iter_plane(plane, anode_id, face_id, plane_id)


def _iter_anode(anode: AnodeGeom, anode_id: int):
    """Yield wire tuples for every wire in *anode*."""
    for face_id, face in enumerate(anode.faces):
        yield from _iter_face(face, anode_id, face_id)


def _iter_detector(det_geom: DetectorGeom):
    """Yield wire tuples for every wire in *det_geom*."""
    for anode_id, anode in enumerate(det_geom.anodes):
        yield from _iter_anode(anode, anode_id)


# ── Public polydata builders ───────────────────────────────────────────────────

def plane_polydata(plane: PlaneGeom, anode_id: int, face_id: int,
                   plane_id: int) -> object:
    """Return a ``vtkPolyData`` for all wires in *plane*."""
    return _build_polydata(_iter_plane(plane, anode_id, face_id, plane_id))


def face_polydata(face: FaceGeom, anode_id: int, face_id: int) -> object:
    """Return a ``vtkPolyData`` for all wires in *face*."""
    return _build_polydata(_iter_face(face, anode_id, face_id))


def anode_polydata(anode: AnodeGeom, anode_id: int) -> object:
    """Return a ``vtkPolyData`` for all wires in *anode*."""
    return _build_polydata(_iter_anode(anode, anode_id))


def detector_polydata(det_geom: DetectorGeom) -> object:
    """Return a ``vtkPolyData`` for all wires in *det_geom*."""
    return _build_polydata(_iter_detector(det_geom))


# ── Multiblock assembly ────────────────────────────────────────────────────────

def _set_block_name(mb, idx: int, name: str) -> None:
    """Set the ParaView-visible name of block *idx* in a multiblock dataset."""
    import vtk
    mb.GetMetaData(idx).Set(vtk.vtkCompositeDataSet.NAME(), name)


def build_multiblock(det_geom: DetectorGeom, blocking: Blocking) -> object:
    """Build a ``vtkMultiBlockDataSet`` with hierarchy matching *blocking*.

    Block hierarchy per level:

    * ``ANODE``  — root → [anode VTP, …]
    * ``FACE``   — root → [anode sub-block → [face VTP, …], …]
    * ``PLANE``  — root → [anode sub-block → [face sub-block →
      [plane VTP, …], …], …]

    Args:
        det_geom: Source geometry.
        blocking: Desired block granularity (``DETECTOR`` is invalid here;
                  use :func:`detector_polydata` directly for that case).

    Returns:
        A nested ``vtkMultiBlockDataSet`` ready for :func:`write_vtm`.
    """
    import vtk

    blocking = Blocking(blocking)
    root = vtk.vtkMultiBlockDataSet()
    root.SetNumberOfBlocks(len(det_geom.anodes))

    for ai, anode in enumerate(det_geom.anodes):
        _set_block_name(root, ai, f"anode_{ai}")

        if blocking == Blocking.ANODE:
            root.SetBlock(ai, anode_polydata(anode, ai))
            continue

        anode_mb = vtk.vtkMultiBlockDataSet()
        anode_mb.SetNumberOfBlocks(len(anode.faces))
        root.SetBlock(ai, anode_mb)

        for fi, face in enumerate(anode.faces):
            _set_block_name(anode_mb, fi, f"face_{fi}")
            sorted_planes = sort_planes_by_drift(face)

            if blocking == Blocking.FACE:
                anode_mb.SetBlock(fi, face_polydata(face, ai, fi))
                continue

            face_mb = vtk.vtkMultiBlockDataSet()
            face_mb.SetNumberOfBlocks(len(sorted_planes))
            anode_mb.SetBlock(fi, face_mb)

            for pi, plane in enumerate(sorted_planes):
                _set_block_name(face_mb, pi, f"plane_{pi}")
                face_mb.SetBlock(pi, plane_polydata(plane, ai, fi, pi))

    return root


# ── Writers ────────────────────────────────────────────────────────────────────

def write_vtp(out_path, polydata) -> pathlib.Path:
    """Write *polydata* to a VTP file, adding ``.vtp`` suffix if absent.

    Returns the path actually written.
    """
    import vtk
    p = pathlib.Path(out_path)
    if p.suffix != ".vtp":
        p = p.with_suffix(".vtp")
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(p))
    writer.SetInputData(polydata)
    writer.Write()
    return p


def write_vtm(out_path, multiblock) -> pathlib.Path:
    """Write *multiblock* to a VTM file, adding ``.vtm`` suffix if absent.

    VTK automatically creates a ``<stem>/`` sub-directory alongside the
    ``.vtm`` file to hold the individual block files.

    Returns the path actually written.
    """
    import vtk
    p = pathlib.Path(out_path)
    if p.suffix != ".vtm":
        p = p.with_suffix(".vtm")
    writer = vtk.vtkXMLMultiBlockDataWriter()
    writer.SetFileName(str(p))
    writer.SetInputData(multiblock)
    writer.Write()
    return p


# ── Main entry point ───────────────────────────────────────────────────────────

def save(filename, detectorgeom: DetectorGeom,
         blocking: Blocking = Blocking.ANODE) -> pathlib.Path:
    """Save wire geometry to a VTK file.

    Args:
        filename:     Output path.  Any ``.vtp`` or ``.vtm`` suffix is stripped
                      and replaced with the correct one for the chosen
                      *blocking*.  For multiblock output, VTK places block
                      files in a ``<stem>/`` sub-directory beside the ``.vtm``.
        detectorgeom: Wire geometry source (e.g. from
                      :func:`~.geom.togeom` or the GDML pipeline).
        blocking:     :class:`Blocking` value or its string equivalent:

                      * ``"detector"`` — single ``.vtp``, all wires flat.
                      * ``"anode"``    — ``.vtm``, one VTP per anode
                        *(default)*.
                      * ``"face"``     — ``.vtm``, anode → face VTP tree.
                      * ``"plane"``    — ``.vtm``, anode → face → plane
                        VTP tree.

    Per-cell data arrays present on every wire:

    =============  =======  ================================================
    Name           Type     Description
    =============  =======  ================================================
    ``anode_id``   int      Anode positional index in the DetectorGeom.
    ``face_id``    int      Face positional index within the anode.
    ``plane_id``   int      Plane positional index in drift order (0=U).
    ``wire_id``    int      Wire index in pitch order (same as ``wip``).
    ``channel_id`` int      Electronics channel (``-1`` when unassigned).
    ``segment``    int      Distance from channel input (0 = nearest).
    ``wip``        int      Wire-in-plane pitch-order index.
    ``tail``       float[3] World-frame endpoint furthest from electronics.
    ``head``       float[3] World-frame endpoint closest to electronics.
    ``arrow``      float[3] Vector ``head − tail`` (active VTK vectors).
    =============  =======  ================================================

    Returns:
        :class:`pathlib.Path` of the file written (``.vtp`` or ``.vtm``).
    """
    blocking = Blocking(blocking)

    p = pathlib.Path(filename)
    stem = p.with_suffix("") if p.suffix in (".vtp", ".vtm") else p

    if blocking == Blocking.DETECTOR:
        return write_vtp(stem, detector_polydata(detectorgeom))

    return write_vtm(stem, build_multiblock(detectorgeom, blocking))
