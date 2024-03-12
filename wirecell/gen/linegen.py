import json
import math
import numpy as np
import dataclasses
from typing import Type
from wirecell.util.wires import array as warray
from wirecell.util.wires import persist as wpersist
from wirecell import units
from wirecell.util.functions import unitify

ENCODING = 'utf-8'

@dataclasses.dataclass
class TrackConfig:
    '''
    Collect the configuration parameters for the track.
    '''

    length: float = 0.0
    t0: float = 0.0
    eperstep: float = 5000.0/units.mm
    step_size: float = 1.0*units.mm
    track_speed: float = unitify("clight")
    theta_y: float = unitify("90*deg")
    theta_xz: float = unitify("45*deg")
    global_angles: bool = True

    @classmethod
    def from_dict(cls: Type["TrackConfig"], obj: dict = {}):
        dat = {f.name:obj.get(f.name, f.default)
               for f in dataclasses.fields(cls)}
        return cls(**dat)

    def to_dict(self):
        return dataclasses.asdict(self)

def generate_line_track_depos(p0, p1, t0, eperstep, step_size, track_speed):
    """Generate a linear track between points p0 and p1 starting at time t0.

    The track is a collection of charge depositions between points p0 and p1.

    Parameters
    ----------
    p0 : np.ndarray, (x0, y0, z0)
        starting location
    p1 : np.ndarray, (x1, y1, z1)
        ending location
    t0 : float
        starting time
    eperstep : float
        number of electrons deposited per step
    step_size : float
        distance between electron depositions
    track_speed : float
        particle speed

    Returns
    -------
    times : np.ndarray, shape (N,)
        times of charge despotition
    points : np.ndarray, shape (N,3)
        locations of charge despotition
    charges : np.ndarray, shape (N,)
        amount of charge deposited at each location
    """
    # pylint: disable=too-many-arguments

    vector    = (p1 - p0)
    distance  = math.sqrt(np.dot(vector, vector))

    n_steps = int(round(distance / step_size))
    t1 = t0 + n_steps * step_size / track_speed

    times   = np.linspace(t0, t1, n_steps+1, endpoint = True)
    points  = np.linspace(p0, p1, n_steps+1, endpoint = True)
    charges = np.full((n_steps+1,), -eperstep)

    return (times, points, charges)

def midpoint_length_direction_to_endpoints(p_mid, length, direction):
    halflen = length / 2

    p0 = p_mid - direction * halflen
    p1 = p_mid + direction * halflen

    return (p0, p1)

def tpc_angles_to_direction(theta_y, theta_xz):
    """Convert TPC angles (theta_y, theta_xz) to dir (cos_x, cos_y, cos_z)

    TPC Angles are defined in Figure 8 of https://arxiv.org/pdf/1802.08709.pdf
    """
    cos_y = math.cos(theta_y)
    cos_z = math.sin(theta_y) * math.cos(theta_xz)
    cos_x = math.sin(theta_y) * math.sin(theta_xz)

    return np.array([ cos_x, cos_y, cos_z ])

def direction_to_tpc_angles(direction):
    """Convert dir (cos_x, cos_y, cos_z) to TPC angles (theta_y, theta_xz)

    TPC Angles are defined in Figure 8 of https://arxiv.org/pdf/1802.08709.pdf

    Note
    ----
        Angles are returned mod 180 degress.

    See Also
    --------
        tpc_angles_to_direction
    """
    theta_y  = math.acos(direction[1])
    theta_xz = math.acos(np.clip(direction[2] / math.sqrt(1 - direction[1]**2), -1, 1))

    return theta_y, theta_xz

def plane_yy_to_rotation_matrix(plane_yy):
    # c.f. Figure 8 of https://arxiv.org/pdf/1802.08709.pdf
    return np.array([
        [1,                   0,                  0],
        [0,  math.cos(plane_yy), math.sin(plane_yy)],
        [0, -math.sin(plane_yy), math.cos(plane_yy)],
    ])

def wp_direction_to_global_direction(R, dir_wp):
    return R.T @ dir_wp

def global_direction_to_wp_direction(R, dir_glb):
    return R @ dir_glb

def pack_track_data(times, points, charges, start_idx = 0):
    # times   : (N, )
    # points  : (N, 3)
    # charges : (N, )
    indices = np.arange(start_idx, start_idx + len(times))

    # data : (N, 7)
    data         = np.zeros((len(times), 7), dtype = 'float32')
    data[:, 0]   = times
    data[:, 1]   = charges
    data[:, 2:5] = points

    # data : (N, 4)
    info       = np.zeros((len(times), 4), dtype = 'int32')
    info[:, 0] = indices

    return (data, info)

def generate_line_track_depo_set(p0, p1, t0, eperstep, step_size, track_speed):
    # pylint: disable=too-many-arguments
    (times, points, charges) = generate_line_track_depos(
        p0, p1, t0, eperstep, step_size, track_speed
    )

    (data, info) = pack_track_data(times, points, charges, start_idx = 0)

    depo_sets = { 'depo_data_0' : data, 'depo_info_0' : info }

    return depo_sets, times, points, charges


ArrayLike = np.ndarray[int, np.dtype[float]]

@dataclasses.dataclass
class TrackMetadata:
    '''
    The metadta about one track.
    '''

    p0: ArrayLike = np.zeros((3,))
    'The starting 3-point of the track.'

    p1: ArrayLike = np.zeros((3,))
    'The ending 3-point of the track.'

    t0: float = 0.0
    'The the time at p0.'
    
    t1: float = 0.0
    'The the time at p1.'

    R_wps: ArrayLike = np.zeros((3,3,3))
    'The plane coordinate rotation matrices as 3(nplanes)x3x3 array.'

    dir_glb: ArrayLike = np.zeros((3,))
    'The 3-vector of the track direction in global coordinates.'

    theta_y_glb: float = 0
    'The angle between global Y-axis and direction vector.'

    theta_xz_glb: float = 0
    'The angle between global Z-axis and the projection of the direction vector into the X-Z plane.'

    dir_wps: ArrayLike = np.zeros((3,3))
    'Three 3-vectors giving the track direction expressed in each wire-plane coordinate system.'

    theta_y_wps: ArrayLike = np.zeros((3,))
    'The angles between each of three wire-plane Y-axes and the direction vector.'

    theta_xz_wps: ArrayLike = np.array((3,))
    'The angles, one for each wire plane coordinate system between their Z-axis and projection of the direction vector into their X-Z plane.'

    eperstep: float = 0
    'The number of electrons per unit track step.'

    track_speed: float = 0
    'The speed of the track.'

    detector: str = ""
    'The canonical name of the detector'

    apa: int = -1
    'The APA number.'

    plane_idx: int = -1
    'The plane index.'
    
    @classmethod
    def from_dict(cls: Type["TrackMetadata"], obj: dict = {}):
        dat = {f.name:obj.get(f.name, f.default)
               for f in dataclasses.fields(cls)}
        return cls(**dat)

    def to_dict(self):
        dat = dict()
        for f in dataclasses.fields(TrackMetadata):
            mem = getattr(self, f.name)
            if isinstance(mem, np.ndarray):
                mem = mem.tolist()
            dat[f.name] = mem
        return dat

def pack_track_metadata(
    points, times, dir_wps, dir_glb, tconf, R_wps
):
    # pylint: disable=too-many-arguments
    theta_y_glb, theta_xz_glb = direction_to_tpc_angles(dir_glb)

    # 3x2: [(y, xz), ...]
    thetas_wp = np.array([direction_to_tpc_angles(dir_wp) for dir_wp in dir_wps])

    metadata = TrackMetadata(
        p0   = tuple(points[0]),
        p1   = tuple(points[-1]),
        t0   = times[0],
        t1   = times[-1],
        R_wps = R_wps.tolist(),
        dir_glb      = tuple(dir_glb),
        theta_y_glb  = theta_y_glb,
        theta_xz_glb = theta_xz_glb,
        dir_wps      = dir_wps.tolist(),
        theta_y_wps  = thetas_wp[:,0].tolist(),
        theta_xz_wps = thetas_wp[:,1].tolist(),
        eperstep     = tconf.eperstep,
        track_speed  = tconf.track_speed,
    )

    return metadata

def generate_line_track(center, tconf, R_wps, plane_idx):
    # pylint: disable=too-many-locals
    if tconf.global_angles:
        dir_glb = tpc_angles_to_direction(tconf.theta_y, tconf.theta_xz)
    else:
        dir_wp  = tpc_angles_to_direction(tconf.theta_y, tconf.theta_xz)
        dir_glb = wp_direction_to_global_direction(R_wps[plane_idx], dir_wp)

    p0, p1 \
        = midpoint_length_direction_to_endpoints(center, tconf.length, dir_glb)

    depo_sets, times, points, _charges = generate_line_track_depo_set(
        p0, p1, tconf.t0, tconf.eperstep, tconf.step_size, tconf.track_speed
    )

    dir_wps = np.array([global_direction_to_wp_direction(R_wp, dir_glb) for R_wp in R_wps])

    metadata = pack_track_metadata(points, times, dir_wps, dir_glb, tconf, R_wps)

    return (depo_sets, metadata)

def generate_and_save_line_track(
    center, track_config, phi, path_depo, path_meta, plane_idx=0
):
    # pylint: disable=too-many-locals
    R_wps = np.array([plane_yy_to_rotation_matrix(p) for p in phi])

    (depo_sets, metadata) = generate_line_track(center, track_config, R_wps, plane_idx)

    np.savez(path_depo, **depo_sets)

    if path_meta is None:
        return

    with open(path_meta, "wt", encoding = ENCODING) as f:
        json.dump(dataclasses.asdict(metadata), f, indent = 4)

def load_wp_spec(detector, apa_idx):
    """
    Load wire plane centers and rotation matrices for a given `detector`.

    First axis of each returned array spans the three planes.
    """
    store = wpersist.load(detector)

    wp_centers = list()
    wp_rots = list()

    # warr : (N, 2, 3)
    #        (wire_idx, [start, end], coord_idx)
    for plane_idx in range(3):
        warr = warray.endpoints_from_schema(
            store, plane = plane_idx, anode = apa_idx
        )
        warr = warray.correct_endpoint_array(warr)

        mean_wire, mean_pitch = warray.mean_wire_pitch(warr)

        wp_rot = warray.rotation(mean_wire, mean_pitch)

        mid_idx   = warr.shape[0] // 2
        wp_center = 0.5 * (warr[mid_idx, 0, ...] + warr[mid_idx, 1, ...])
        wp_centers.append(wp_center)
        wp_rots.append(wp_rot)

    return (np.array(wp_centers), np.array(wp_rots))

def generate_and_save_line_track_in_detector(
    detector, apa_idx, plane_idx, offset, track_config, path_depo, path_meta
):
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    wp_centers, R_wps = load_wp_spec(detector, apa_idx)
    center = wp_centers[plane_idx] + offset

    (depo_sets, metadata) = generate_line_track(center, track_config, R_wps, plane_idx)

    np.savez(path_depo, **depo_sets)

    if path_meta is None:
        return

    metadata.detector  = detector
    metadata.apa       = apa_idx
    metadata.plane_idx = plane_idx

    with open(path_meta, "wt", encoding = ENCODING) as f:
        json.dump(dataclasses.asdict(metadata), f, indent = 4)

