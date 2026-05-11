from dataclasses import dataclass, field
import numpy as np
from typing import Optional, Union

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


