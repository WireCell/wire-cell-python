from collections import namedtuple
import json
import math
import numpy as np

from wirecell.util.wires import array as warray
from wirecell.util.wires import persist as wpersist

ENCODING = 'utf-8'

TrackConfig = namedtuple('TrackConfig', [
    'length',
    't0',
    'eperstep',
    'step_size',
    'track_speed',
    'theta_y',
    'theta_xz',
    'global_angles'
])

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
    theta_xz = math.acos(direction[2] / math.sqrt(1 - direction[1]**2))

    return theta_y, theta_xz

def plane_yy_to_rotation_matrix(plane_yy):
    # c.f. Figure 8 of https://arxiv.org/pdf/1802.08709.pdf
    return np.array([
        [1,                   0,                  0],
        [0,  math.cos(plane_yy), math.sin(plane_yy)],
        [0, -math.sin(plane_yy), math.cos(plane_yy)],
    ])

def wp_direction_to_global_direction(R, dir_wp):
    return R.T() @ dir_wp

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

def pack_track_metadata(
    points, times, dir_wp, dir_glb, tconf, R_wp
):
    # pylint: disable=too-many-arguments
    theta_y_glb, theta_xz_glb = direction_to_tpc_angles(dir_glb)
    theta_y_wp, theta_xz_wp   = direction_to_tpc_angles(dir_wp)

    metadata = {
        'p0'   : tuple(points[0]),
        'p1'   : tuple(points[-1]),
        't0'   : times[0],
        't1'   : times[-1],
        'R_wp' : R_wp.tolist(),
        'dir_glb'      : tuple(dir_glb),
        'theta_y_glb'  : math.degrees(theta_y_glb),
        'theta_xz_glb' : math.degrees(theta_xz_glb),
        'dir_wp'       : tuple(dir_wp),
        'theta_y_wp'   : math.degrees(theta_y_wp),
        'theta_xz_wp'  : math.degrees(theta_xz_wp),
        'eperstep'     : tconf.eperstep,
        'track_speed'  : tconf.track_speed,
    }

    return metadata

def generate_line_track(center, tconf, R_wp):
    # pylint: disable=too-many-locals

    if tconf.global_angles:
        dir_glb = tpc_angles_to_direction(tconf.theta_y, tconf.theta_xz)
        dir_wp  = global_direction_to_wp_direction(R_wp, dir_glb)
    else:
        dir_wp  = tpc_angles_to_direction(tconf.theta_y, tconf.theta_xz)
        dir_glb = wp_direction_to_global_direction(R_wp, dir_wp)

    p0, p1 \
        = midpoint_length_direction_to_endpoints(center, tconf.length, dir_glb)

    depo_sets, times, points, _charges = generate_line_track_depo_set(
        p0, p1, tconf.t0, tconf.eperstep, tconf.step_size, tconf.track_speed
    )

    metadata = pack_track_metadata(points, times, dir_wp, dir_glb, tconf, R_wp)

    return (depo_sets, metadata)

def generate_and_save_line_track(
    center, track_config, phi, path_depo, path_meta
):
    # pylint: disable=too-many-locals
    R_wp = plane_yy_to_rotation_matrix(phi)

    (depo_sets, metadata) = generate_line_track(center, track_config, R_wp)

    np.savez(path_depo, **depo_sets)

    if path_meta is None:
        return

    with open(path_meta, "wt", encoding = ENCODING) as f:
        json.dump(metadata, f, indent = 4)

def load_wp_spec(detector, apa_idx, plane_idx):
    """Load wire plane center and rotation matrix for a given `detector`"""
    store = wpersist.load(detector)

    # warr : (N, 2, 3)
    #        (wire_idx, [start, end], coord_idx)
    warr = warray.endpoints_from_schema(
        store, plane = plane_idx, anode = apa_idx
    )
    warr = warray.correct_endpoint_array(warr)

    mean_wire, mean_pitch = warray.mean_wire_pitch(warr)

    wp_rot = warray.rotation(mean_wire, mean_pitch)

    mid_idx   = warr.shape[0] // 2
    wp_center = 0.5 * (warr[mid_idx, 0, ...] + warr[mid_idx, 1, ...])

    return (wp_center, wp_rot)

def generate_and_save_line_track_in_detector(
    detector, apa_idx, plane_idx, offset, track_config, path_depo, path_meta
):
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    wp_center, R_wp = load_wp_spec(detector, apa_idx, plane_idx)
    center = wp_center + offset

    (depo_sets, metadata) = generate_line_track(center, track_config, R_wp)

    np.savez(path_depo, **depo_sets)

    if path_meta is None:
        return

    metadata['detector']  = detector
    metadata['apa']       = apa_idx
    metadata['plane_idx'] = plane_idx

    with open(path_meta, "wt", encoding = ENCODING) as f:
        json.dump(metadata, f, indent = 4)

