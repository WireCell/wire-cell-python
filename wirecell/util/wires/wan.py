'''
Wire Attachment Number (WAN) ordering utilities.

WAN orders the channels within a plane by the pitch position of their
segment-0 wire (the wire directly connected to the electronics).  Increasing
WAN corresponds to increasing pitch position, where pitch direction is

    p̂ = x̂ × ŵ

with x̂ = [1, 0, 0] (drift direction) and ŵ the average wire direction
(tail → head) computed from the segment-0 wires in the plane.

Functions operate on the dict-based representation produced by
``wirecell.util.wires.info.todict()``.
'''

import numpy


def plane(p):
    '''Return a list of channel IDs (CHIDs) for a plane, ordered by
    increasing pitch position (WAN order).

    Parameters
    ----------
    p : dict
        One plane object from the list returned by ``info.todict()``, i.e.::

            {"ident": <int>, "wires": [{"ident": ..., "channel": ...,
                                        "segment": ...,
                                        "head": {"x":, "y":, "z":},
                                        "tail": {"x":, "y":, "z":}}, ...]}

    Returns
    -------
    list of int
        Channel IDs ordered by increasing pitch (WAN 0, 1, 2, …).
    '''
    seg0 = [w for w in p['wires'] if w['segment'] == 0]
    if not seg0:
        return []

    def wire_vec(w):
        h = w['head']
        t = w['tail']
        return numpy.array([h['x'] - t['x'], h['y'] - t['y'], h['z'] - t['z']])

    def wire_center(w):
        h = w['head']
        t = w['tail']
        return numpy.array([(h['x'] + t['x']) / 2,
                             (h['y'] + t['y']) / 2,
                             (h['z'] + t['z']) / 2])

    # Average wire direction from segment-0 wires.
    vecs = numpy.array([wire_vec(w) for w in seg0])
    avg_vec = vecs.mean(axis=0)
    norm = numpy.linalg.norm(avg_vec)
    if norm == 0:
        raise ValueError(f"plane {p['ident']}: degenerate average wire direction")
    avg_vec /= norm

    # Pitch direction: p̂ = x̂ × ŵ
    xhat = numpy.array([1.0, 0.0, 0.0])
    pdir = numpy.cross(xhat, avg_vec)
    pnorm = numpy.linalg.norm(pdir)
    if pnorm == 0:
        raise ValueError(f"plane {p['ident']}: wire direction parallel to drift — pitch direction undefined")
    pdir /= pnorm

    # Sort by projection onto pitch direction.
    centers = numpy.array([wire_center(w) for w in seg0])
    pitches = centers.dot(pdir)

    order = numpy.argsort(pitches)
    return [seg0[i]['channel'] for i in order]


def face(f):
    '''Return a mapping from plane ident to WAN-ordered CHID list for a face.

    Parameters
    ----------
    f : dict
        One face object from the list returned by ``info.todict()``, i.e.::

            {"ident": <int>, "planes": [<plane dict>, ...]}

    Returns
    -------
    dict
        ``{plane_ident (int): [chid, chid, ...], ...}`` where each list is
        in WAN order (increasing pitch).
    '''
    return {p['ident']: plane(p) for p in f['planes']}


def anode_channels(a):
    '''Return the set of channel IDs owned by an anode (segment-0 wires only).

    Parameters
    ----------
    a : dict
        One anode object from ``info.todict()``.

    Returns
    -------
    set of int
    '''
    chids = set()
    for f in a['faces']:
        for p in f['planes']:
            for w in p['wires']:
                if w['segment'] == 0:
                    chids.add(w['channel'])
    return chids


def anode_partition(detector, chids):
    '''Partition a collection of channel IDs by which anode provides them.

    Parameters
    ----------
    detector : dict
        One detector object from ``info.todict()``, i.e.::

            {"ident": <int>, "anodes": [<anode dict>, ...]}

    chids : iterable of int
        Channel IDs to partition.

    Returns
    -------
    dict
        ``{anode_ident (int): set_of_chids}`` — each value is the subset of
        *chids* whose segment-0 wires belong to that anode.  Anodes that
        share no channel with *chids* are omitted from the result.  A channel
        that does not appear in any anode is silently dropped.
    '''
    wanted = set(chids)
    result = {}
    for a in detector['anodes']:
        owned = anode_channels(a) & wanted
        if owned:
            result[a['ident']] = owned
    return result


def anode_chids(a, det=None):
    '''Return a flat, WAN-ordered list of all CHIDs for an anode.

    The ordering is deterministic and hierarchical:

    * Faces are visited in ``anode_faces(a, det)`` order (natural storage order
      by default).
    * Within each face, planes are visited in ascending plane-ident order
      (0 = U induction, 1 = V induction, 2 = W/X collection).
    * Within each plane, channels are in WAN pitch order (``plane()``).

    This list defines the canonical row layout used by ``frames.wan_pad``:
    row *i* in the output array belongs to ``anode_chids(a)[i]``.

    Parameters
    ----------
    a : dict
        One anode object from ``info.todict()``.
    det : ignored
        Passed through to ``anode_faces()``; reserved for future face-ordering
        changes.

    Returns
    -------
    list of int
        All CHIDs owned by the anode (segment-0 wires only) in WAN order.
    '''
    face_by_ident = {f['ident']: f for f in a['faces']}
    chids = []
    for fid in anode_faces(a, det=det):
        face_planes = face(face_by_ident[fid])       # {plane_ident: [chid, ...]}
        for pid in sorted(face_planes.keys()):
            chids.extend(face_planes[pid])
    return chids


def anode_faces(a, det=None):
    '''Return an ordered list of face ident numbers for an anode.

    Parameters
    ----------
    a : dict
        One anode object from the list returned by ``info.todict()``, i.e.::

            {"ident": <int>, "faces": [<face dict>, ...]}
    det : ignored
        Reserved for future use (e.g. detector-specific face ordering).
        Currently has no effect; faces are returned in their natural order.

    Returns
    -------
    list of int
        Face ident numbers in natural (storage) order.
    '''
    return [f['ident'] for f in a['faces']]
