"""Compatibility shim: re-exports from wirecell.util.wires.gdml and .geom.

The canonical implementation lives in wirecell.util.wires.  This module
exists so that legacy imports of the form ``from wirecell.util.gdml import
…`` continue to work after the move.
"""

from wirecell.util.wires.gdml import *   # noqa: F401,F403
from wirecell.util.wires.gdml import (   # explicit re-export for linters
    BUILTIN_CONFIGS,
    load_config,
    build_store,
    build_hd_channel_map,
    convert,
    match_role,
    classify_volumes,
    parse_define,
    parse_solids,
    parse_structure,
    extract_wires,
    build_detector_faces,
    pair_faces_into_anodes,
    assign_vd_channels,
    find_vd_connected_pairs,
    gdml_transform,
    compose_transforms,
    apply_transform,
)
from wirecell.util.wires.geom import (   # noqa: F401
    WireGeom,
    PlaneGeom,
    FaceGeom,
    AnodeGeom,
    DetectorGeom,
    sort_wires_by_pitch,
    sort_planes_by_drift,
    togeom,
    tostore,
)
