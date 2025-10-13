import bz2
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import torch

# ====================================================================
# 1. PYTHON DATA STRUCTURES (Replacing C++ structs/classes)
# ====================================================================

def get_center(wire, drift='vd'):
    center =  torch.mean(torch.Tensor([
            [wire.head.z, wire.head.y],
            [wire.tail.z, wire.tail.y],
    ]), dim=0)
    return center

def views_from_schema(store, face_index, drift='vd'):
    #GEt the plane objects from the store for htis face
    planes = [store.planes[i] for i in store.faces[face_index].planes]


    views = []
    #For each plane, get the first and second wire to get the pitch direction and magnitude
    for plane in planes:
        first_wire = store.wires[plane.wires[0]]
        second_wire = store.wires[plane.wires[1]]
        # print(first_wire, second_wire)

        first_center = get_center(first_wire)
        # print(first_center)

        second_center = get_center(second_wire)
        # print(second_center)
        views.append(torch.cat([first_center.unsqueeze(0), second_center.unsqueeze(0)], dim=0).unsqueeze(0))

    return torch.cat(views)

@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

@dataclass
class Wire:
    ident: int = 0
    channel: int = 0
    segment: int = 0
    tail: Optional[Point] = None  # Will be set to an object from the points list
    head: Optional[Point] = None  # Will be set to an object from the points list

@dataclass
class Plane:
    ident: int = 0
    wires: List[int] = field(default_factory=list) # List of wire indices/idents

@dataclass
class Face:
    ident: int = 0
    planes: List[int] = field(default_factory=list) # List of plane indices/idents

@dataclass
class Anode:
    ident: int = 0
    faces: List[int] = field(default_factory=list) # List of face indices/idents

@dataclass
class Detector:
    ident: int = 0
    anodes: List[int] = field(default_factory=list) # List of anode indices/idents

@dataclass
class StoreDB:
    """The main container for all loaded geometry data."""
    wires: List[Wire] = field(default_factory=list)
    planes: List[Plane] = field(default_factory=list)
    faces: List[Face] = field(default_factory=list)
    anodes: List[Anode] = field(default_factory=list)
    detectors: List[Detector] = field(default_factory=list)


# ====================================================================
# 2. HELPER FUNCTIONS
# ====================================================================

def load_bz2_json(path: str) -> Dict[str, Any]:
    """
    Abstracted function to load a JSON file, handling bz2 compression.
    This replaces WireCell::Persist::load(path) and handles the "jtop" load.
    """
    # 'rt' mode opens the file in read text mode, automatically decompressing
    # the bz2 content and decoding the bytes to a string.
    with bz2.open(path, 'rt', encoding='utf-8') as bz_file:
        jtop = json.load(bz_file)
    return jtop

def load_file(path: str, store: StoreDB):
    """
    Converts the C++ load_file logic to Python.
    Parses geometry data from a JSON structure and populates the StoreDB object.
    """

    # Abstraction of WireCell::Persist::load(path)
    jtop = load_bz2_json(path)
    
    # Access the main "Store" object
    jstore = jtop.get("Store", {})

    # The C++ code uses a helper 'get' function; Python uses dict.get()
    # Note: Python uses 0-based indexing, C++ uses 0-based indexing for vectors.
    
    # ----------------------------------------------------------------
    # 1. Points (Temporary storage, as they are referenced by wires)
    # ----------------------------------------------------------------
    points: List[Point] = []
    jpoints = jstore.get("points", [])
    
    for jp_wrapper in jpoints:
        # Assumes the structure is an array of {"Point": { "x": ..., "y": ...}}
        jp = jp_wrapper.get("Point", {}) 
        
        point = Point(
            x=jp.get("x", 0.0),
            y=jp.get("y", 0.0),
            z=jp.get("z", 0.0)
        )
        points.append(point)

    # ----------------------------------------------------------------
    # 2. Wires
    # ----------------------------------------------------------------
    jwires = jstore.get("wires", [])
    
    for jw_wrapper in jwires:
        jwire = jw_wrapper.get("Wire", {})
        
        wire = Wire(
            ident=jwire.get("ident", 0),
            channel=jwire.get("channel", 0),
            segment=jwire.get("segment", 0)
        )
        
        # Look up Points using the indices 'tail' and 'head'
        itail = jwire.get("tail")
        ihead = jwire.get("head")
        
        # Access the Point objects by index (assuming indices are valid)
        if itail is not None and 0 <= itail < len(points):
            wire.tail = points[itail]
        if ihead is not None and 0 <= ihead < len(points):
            wire.head = points[ihead]
            
        store.wires.append(wire)

    # ----------------------------------------------------------------
    # 3. Planes
    # ----------------------------------------------------------------
    jplanes = jstore.get("planes", [])
    
    for jp_wrapper in jplanes:
        jplane = jp_wrapper.get("Plane", {})
        
        plane = Plane(
            ident=jplane.get("ident", 0),
            # The 'wires' field is an array of ints in the JSON
            wires=jplane.get("wires", []) 
        )
        store.planes.append(plane)

    # ----------------------------------------------------------------
    # 4. Faces
    # ----------------------------------------------------------------
    jfaces = jstore.get("faces", [])
    
    for jf_wrapper in jfaces:
        jface = jf_wrapper.get("Face", {})
        
        face = Face(
            ident=jface.get("ident", 0),
            # The 'planes' field is an array of ints in the JSON
            planes=jface.get("planes", []) 
        )
        store.faces.append(face)

    # ----------------------------------------------------------------
    # 5. Anodes
    # ----------------------------------------------------------------
    janodes = jstore.get("anodes", [])
    
    for ja_wrapper in janodes:
        janode = ja_wrapper.get("Anode", {})
        
        anode = Anode(
            ident=janode.get("ident", 0),
            # The 'faces' field is an array of ints in the JSON
            faces=janode.get("faces", [])
        )
        store.anodes.append(anode)

    # ----------------------------------------------------------------
    # 6. Detectors (Conditional Logic)
    # ----------------------------------------------------------------
    jdets = jstore.get("detectors", [])
    ndets = len(jdets)

    if not ndets:
        # If no detectors are found, create a default detector containing all anodes
        det = Detector(ident=0)
        num_anodes = len(store.anodes)
        det.anodes = list(range(num_anodes)) # 0, 1, 2, ...
        store.detectors.append(det)
    else:
        for jd_wrapper in jdets:
            jdet = jd_wrapper.get("Detector", {})
            
            detector = Detector(
                ident=jdet.get("ident", 0),
                # The 'anodes' field is an array of ints in the JSON
                anodes=jdet.get("anodes", [])
            )
            store.detectors.append(detector)


# ====================================================================
# 3. MOCK USAGE EXAMPLE (to demonstrate functionality)
# ====================================================================

# NOTE: You would replace this section with your actual file path
# db = StoreDB()
# try:
#     load_file("your_geometry_file.json.bz2", db)
#     print(f"Successfully loaded {len(db.wires)} wires and {len(db.detectors)} detectors.")
# except FileNotFoundError:
#     print("Please create a mock or provide a real .json.bz2 file for testing.")