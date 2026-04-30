#!/usr/bin/env python
'''Tests for wirecell.util.wires.wan'''

import pytest
import numpy

from wirecell.util.wires import wan


# ---------------------------------------------------------------------------
# Helpers to build minimal synthetic wire-schema dicts
# ---------------------------------------------------------------------------

def make_wire(ident, channel, segment, tail, head):
    '''Build a wire dict as returned by info.todict().'''
    return dict(
        ident=ident,
        channel=channel,
        segment=segment,
        tail=dict(x=tail[0], y=tail[1], z=tail[2]),
        head=dict(x=head[0], y=head[1], z=head[2]),
    )


def make_plane(ident, wires):
    return dict(ident=ident, wires=wires)


def make_face(ident, planes):
    return dict(ident=ident, planes=planes)


def make_anode(ident, faces):
    return dict(ident=ident, faces=faces)


# ---------------------------------------------------------------------------
# Planes with wires parallel to Y axis (ŵ ≈ ŷ)
#   p̂ = x̂ × ŷ = [1,0,0] × [0,1,0] = [0,0,1]   → pitch increases with +Z
# ---------------------------------------------------------------------------

def y_wire(channel, segment, z_center):
    '''Wire parallel to Y at z=z_center, centered on y=0.'''
    return make_wire(channel, channel, segment,
                     tail=[0.0, -1.0, z_center],
                     head=[0.0,  1.0, z_center])


def make_y_plane(z_positions, channels=None, extra_wires=None):
    '''Plane of segment-0 Y-parallel wires at given Z positions.'''
    if channels is None:
        channels = list(range(len(z_positions)))
    wires = [y_wire(ch, 0, z) for ch, z in zip(channels, z_positions)]
    if extra_wires:
        wires.extend(extra_wires)
    return make_plane(0, wires)


# ---------------------------------------------------------------------------
# wan.plane() tests
# ---------------------------------------------------------------------------

def test_plane_order_increasing():
    '''Channels sorted by increasing Z when wires run along Y.'''
    # z positions deliberately out of order
    z = [3.0, 1.0, 4.0, 2.0]
    p = make_y_plane(z, channels=[30, 10, 40, 20])
    result = wan.plane(p)
    # Expected: sorted by z → channels [10, 20, 30, 40]
    assert result == [10, 20, 30, 40]


def test_plane_order_increasing_negative_z():
    '''Works with negative pitch coordinates.'''
    z = [-1.0, -3.0, 0.0, -2.0]
    p = make_y_plane(z, channels=[0, 1, 2, 3])
    result = wan.plane(p)
    # sorted by z: -3, -2, -1, 0 → channels 1, 3, 0, 2
    assert result == [1, 3, 0, 2]


def test_plane_segment0_only():
    '''Segment > 0 wires are excluded from the WAN ordering.'''
    z_seg0 = [1.0, 3.0, 2.0]
    # channels 0,1,2 are segment 0; channel 99 is segment 1 at z=0 (would sort first if included)
    wires_seg0 = [y_wire(ch, 0, z) for ch, z in zip([0, 1, 2], z_seg0)]
    intruder = y_wire(99, 1, 0.0)  # segment=1, z=0 → would be first if counted
    p = make_plane(0, wires_seg0 + [intruder])
    result = wan.plane(p)
    assert 99 not in result
    assert result == [0, 2, 1]   # sorted by z: 1.0→ch0, 2.0→ch2, 3.0→ch1


def test_plane_single_wire():
    '''Single wire returns a one-element list.'''
    p = make_y_plane([5.0], channels=[42])
    assert wan.plane(p) == [42]


def test_plane_empty_segment0():
    '''Plane with no segment-0 wires returns empty list.'''
    intruder = y_wire(7, 1, 0.0)
    p = make_plane(0, [intruder])
    assert wan.plane(p) == []


def test_plane_pitch_direction_z_wires():
    '''Wires along Z axis: ŵ = ẑ, p̂ = x̂ × ẑ = [0,-1,0] → pitch decreases with Y.'''
    # wire parallel to Z at y=y_center
    def z_wire(channel, y_center):
        return make_wire(channel, channel, 0,
                         tail=[0.0, y_center, -1.0],
                         head=[0.0, y_center,  1.0])

    # channels 0,1,2 at y = 3, 1, 2 respectively
    wires = [z_wire(ch, y) for ch, y in zip([0, 1, 2], [3.0, 1.0, 2.0])]
    p = make_plane(0, wires)
    result = wan.plane(p)
    # p̂ = [0,-1,0] → pitch = -y → sorted by -y: y=3→-3 smallest, y=2→-2, y=1→-1 largest
    # so order by increasing pitch: ch0 (y=3, pitch=-3), ch2 (y=2, pitch=-2), ch1 (y=1, pitch=-1)
    assert result == [0, 2, 1]


def test_plane_unique_channels():
    '''No duplicate channel IDs in the result.'''
    z = [float(i) for i in range(10)]
    p = make_y_plane(z)
    result = wan.plane(p)
    assert len(result) == len(set(result))


# ---------------------------------------------------------------------------
# wan.face() tests
# ---------------------------------------------------------------------------

def make_two_plane_face():
    '''A face with two planes (idents 0 and 1), Y-parallel wires.'''
    p0 = make_y_plane([2.0, 1.0, 3.0], channels=[20, 10, 30])
    p1_obj = make_plane(1, [y_wire(ch, 0, z) for ch, z in zip([5, 6, 4], [5.0, 6.0, 4.0])])
    return make_face(0, [p0, p1_obj])


def test_face_returns_dict():
    f = make_two_plane_face()
    result = wan.face(f)
    assert isinstance(result, dict)


def test_face_keys_are_plane_idents():
    f = make_two_plane_face()
    result = wan.face(f)
    assert set(result.keys()) == {0, 1}


def test_face_values_are_ordered():
    f = make_two_plane_face()
    result = wan.face(f)
    # plane 0: z=[2,1,3] → sorted: ch10, ch20, ch30
    assert result[0] == [10, 20, 30]
    # plane 1: z=[5,6,4] → sorted: ch4, ch5, ch6
    assert result[1] == [4, 5, 6]


def test_face_single_plane():
    p = make_y_plane([1.0, 0.0], channels=[7, 8])
    f = make_face(3, [p])
    result = wan.face(f)
    assert list(result.keys()) == [0]
    assert result[0] == [8, 7]


# ---------------------------------------------------------------------------
# wan.anode_faces() tests
# ---------------------------------------------------------------------------

def make_anode_two_faces():
    faces = [make_face(0, []), make_face(1, [])]
    return make_anode(0, faces)


def test_anode_faces_returns_list():
    a = make_anode_two_faces()
    result = wan.anode_faces(a)
    assert isinstance(result, list)


def test_anode_faces_idents():
    a = make_anode_two_faces()
    assert wan.anode_faces(a) == [0, 1]


def test_anode_faces_natural_order():
    '''anode_faces preserves the storage order of faces.'''
    faces = [make_face(3, []), make_face(1, []), make_face(2, [])]
    a = make_anode(0, faces)
    assert wan.anode_faces(a) == [3, 1, 2]


def test_anode_faces_det_ignored():
    '''det parameter does not change the output (reserved for future use).'''
    a = make_anode_two_faces()
    assert wan.anode_faces(a, det=None) == wan.anode_faces(a, det="pdsp")


def test_anode_faces_single():
    a = make_anode(0, [make_face(5, [])])
    assert wan.anode_faces(a) == [5]


# ---------------------------------------------------------------------------
# Helpers for anode_channels / anode_partition
# ---------------------------------------------------------------------------

def make_detector_two_anodes():
    '''Detector with two anodes owning disjoint channel sets.

    Anode 0: channels 0-5 (two faces, one plane each)
    Anode 1: channels 10-15 (two faces, one plane each)
    '''
    def anode_with_channels(ident, ch_start):
        face0 = make_face(0, [make_plane(0, [y_wire(ch, 0, float(i)) for i, ch in enumerate(range(ch_start, ch_start + 3))])])
        face1 = make_face(1, [make_plane(1, [y_wire(ch, 0, float(i)) for i, ch in enumerate(range(ch_start + 3, ch_start + 6))])])
        return make_anode(ident, [face0, face1])

    return dict(ident=0, anodes=[anode_with_channels(0, 0), anode_with_channels(1, 10)])


# ---------------------------------------------------------------------------
# wan.anode_channels() tests
# ---------------------------------------------------------------------------

def test_anode_channels_returns_set():
    det = make_detector_two_anodes()
    a = det['anodes'][0]
    result = wan.anode_channels(a)
    assert isinstance(result, set)


def test_anode_channels_correct_set():
    det = make_detector_two_anodes()
    assert wan.anode_channels(det['anodes'][0]) == {0, 1, 2, 3, 4, 5}
    assert wan.anode_channels(det['anodes'][1]) == {10, 11, 12, 13, 14, 15}


def test_anode_channels_segment0_only():
    '''Segment > 0 wires are not counted as owned channels.'''
    intruder = y_wire(99, 1, 0.0)   # segment=1
    p = make_plane(0, [y_wire(0, 0, 1.0), intruder])
    a = make_anode(0, [make_face(0, [p])])
    assert wan.anode_channels(a) == {0}


def test_anode_channels_empty_anode():
    a = make_anode(0, [])
    assert wan.anode_channels(a) == set()


# ---------------------------------------------------------------------------
# wan.anode_partition() tests
# ---------------------------------------------------------------------------

def test_anode_partition_returns_dict():
    det = make_detector_two_anodes()
    result = wan.anode_partition(det, [0, 1, 10])
    assert isinstance(result, dict)


def test_anode_partition_basic():
    det = make_detector_two_anodes()
    result = wan.anode_partition(det, [0, 1, 2, 3, 4, 5, 10, 11])
    assert result[0] == {0, 1, 2, 3, 4, 5}
    assert result[1] == {10, 11}


def test_anode_partition_subset():
    '''Only the requested chids appear in the result.'''
    det = make_detector_two_anodes()
    result = wan.anode_partition(det, [1, 3, 11])
    assert result[0] == {1, 3}
    assert result[1] == {11}


def test_anode_partition_empty_input():
    det = make_detector_two_anodes()
    result = wan.anode_partition(det, [])
    assert result == {}


def test_anode_partition_unknown_chids_dropped():
    '''Channel IDs not owned by any anode are silently omitted.'''
    det = make_detector_two_anodes()
    result = wan.anode_partition(det, [0, 999])
    assert 999 not in result.get(0, set())
    assert all(999 not in s for s in result.values())


def test_anode_partition_empty_anode_omitted():
    '''Anodes with no matching channels are absent from the result dict.'''
    det = make_detector_two_anodes()
    # only request channels from anode 0
    result = wan.anode_partition(det, [0, 1, 2])
    assert 1 not in result
    assert 0 in result


def test_anode_partition_all_channels():
    '''Passing all channels covers every anode.'''
    det = make_detector_two_anodes()
    all_ch = list(range(6)) + list(range(10, 16))
    result = wan.anode_partition(det, all_ch)
    assert set(result.keys()) == {0, 1}
    assert result[0] == set(range(6))
    assert result[1] == set(range(10, 16))


def test_anode_partition_values_are_sets():
    det = make_detector_two_anodes()
    result = wan.anode_partition(det, [0, 10])
    for v in result.values():
        assert isinstance(v, set)


# ---------------------------------------------------------------------------
# Integration smoke-test against a real detector file
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pdsp_detector():
    '''Load the full pdsp detector dict.  Skips if not available.'''
    try:
        from wirecell.util.wires import persist as wpersist
        from wirecell.util.wires import info as winfo
        store = wpersist.load("pdsp")
        d = winfo.todict(store)
        return d[0]
    except Exception as exc:
        pytest.skip(f"pdsp wire file not available: {exc}")


@pytest.fixture(scope="module")
def pdsp_anode(pdsp_detector):
    '''Return the first anode of the pdsp detector.'''
    return pdsp_detector['anodes'][0]


def test_pdsp_anode_faces(pdsp_anode):
    fids = wan.anode_faces(pdsp_anode)
    assert len(fids) == 2
    assert len(set(fids)) == len(fids)   # no duplicates


def test_pdsp_face_plane_keys(pdsp_anode):
    for fi, f in enumerate(pdsp_anode['faces']):
        result = wan.face(f)
        assert isinstance(result, dict)
        # PDSP has 3 planes per face (U, V, W)
        assert len(result) == 3


def test_pdsp_plane_channel_count(pdsp_anode):
    '''Each plane returns the expected number of channels (800 for PDSP U/V/W).'''
    for f in pdsp_anode['faces']:
        for p in f['planes']:
            chids = wan.plane(p)
            # PDSP: 800 wires per plane per face
            assert len(chids) == 800, (
                f"plane {p['ident']} returned {len(chids)} channels, expected 800"
            )


def test_pdsp_plane_no_duplicates(pdsp_anode):
    '''No channel appears twice within a plane.'''
    for f in pdsp_anode['faces']:
        for p in f['planes']:
            chids = wan.plane(p)
            assert len(chids) == len(set(chids)), (
                f"plane {p['ident']} has duplicate channel IDs"
            )


def test_pdsp_pitch_strictly_increasing(pdsp_anode):
    '''Pitch projections are strictly increasing (no tied wire centers).'''
    import numpy as np
    from wirecell.util.wires import wan as _wan

    def pitch_positions(p_dict):
        seg0 = [w for w in p_dict['wires'] if w['segment'] == 0]
        vecs = np.array([[w['head'][k] - w['tail'][k] for k in 'xyz'] for w in seg0])
        avg_vec = vecs.mean(axis=0)
        avg_vec /= np.linalg.norm(avg_vec)
        pdir = np.cross([1.0, 0.0, 0.0], avg_vec)
        pdir /= np.linalg.norm(pdir)
        centers = np.array([[(w['head'][k] + w['tail'][k]) / 2 for k in 'xyz'] for w in seg0])
        return centers.dot(pdir)

    for f in pdsp_anode['faces']:
        for p in f['planes']:
            pitches = pitch_positions(p)
            diffs = np.diff(np.sort(pitches))
            assert np.all(diffs > 0), (
                f"plane {p['ident']} has non-strictly-increasing pitch positions"
            )


def test_pdsp_anode_partition_covers_all(pdsp_detector):
    '''anode_partition on the full channel set returns every anode and every channel.'''
    # Collect every channel in the detector.
    all_ch = set()
    for a in pdsp_detector['anodes']:
        all_ch |= wan.anode_channels(a)

    result = wan.anode_partition(pdsp_detector, all_ch)

    # Every anode must appear.
    expected_anode_idents = {a['ident'] for a in pdsp_detector['anodes']}
    assert set(result.keys()) == expected_anode_idents

    # The union of all partitions equals the full channel set.
    union = set()
    for s in result.values():
        union |= s
    assert union == all_ch


def test_pdsp_anode_partition_disjoint(pdsp_detector):
    '''Each channel belongs to exactly one anode in the partition.'''
    all_ch = set()
    for a in pdsp_detector['anodes']:
        all_ch |= wan.anode_channels(a)

    result = wan.anode_partition(pdsp_detector, all_ch)

    seen = set()
    for s in result.values():
        assert seen.isdisjoint(s), "channel appears in more than one anode partition"
        seen |= s


def test_pdsp_anode_partition_subset(pdsp_detector):
    '''Requesting a subset of channels returns only those channels.'''
    # grab a handful of channels from the first anode only
    first_anode = pdsp_detector['anodes'][0]
    sample = list(wan.anode_channels(first_anode))[:10]
    result = wan.anode_partition(pdsp_detector, sample)
    assert set(result.keys()) == {first_anode['ident']}
    assert result[first_anode['ident']] == set(sample)
