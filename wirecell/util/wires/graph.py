#!/usr/bin/env python
'''
Do fun stuff to a connectivity graph
'''

import networkx
from wirecell import units

def nodes_by_type(G, typename):
    '''
    Return a list of all nodes in G of the given type.
    '''
    return [n for n in G.nodes if G.node[n]['type'] == typename]


def neighbors_by_type(G, seed, typename, radius=1):
    '''
    Return a set of all neighbor nodes withing given radius of seed
    which are of the given typename.
    '''
    if radius == 1:
        return set([n for n in networkx.neighbors(G, seed) if G.node[n]['type'] == typename])

    return set([n for n in networkx.ego_graph(G, seed, radius) if G.node[n]['type'] == typename])

def neighbors_by_path(G, seed, typenamepath):
    '''
    Return all neighbor nodes by following a path of type names from
    given seed.
    '''
    if not typenamepath:
        return set()
    nn = neighbors_by_type(G, seed, typenamepath[0])
    for node in list(nn):
        nnn = neighbors_by_path(G, node, typenamepath[1:])
        nn.update(nnn)
    return nn

def child_by_path(G, seed, edgepath):
    '''
    Return a child by walking an "edgepath" from a seed node.

    The edgepath is a list of triples:

    >>> [(typename, attrname, attrval), ...]

    Where `typename` is the node type for the step in the path and
    `attrname` and `attrval` give a name/value for an edge attribute
    which must join current node with next node.
    '''
    if not edgepath:
        return seed

    tn, an, av = edgepath.pop(0)

    for nn in neighbors_by_type(G, seed, tn):
        try:
            val = G[seed][nn][an]
        except KeyError:
            continue
        if val != av:
            continue
        return child_by_path(G, nn, edgepath)
    return
            

def parent(G, child, parent_type):
    '''
    Return parent node of given type 
    '''
    for n in networkx.neighbors(G, child):
        if G.node[n]['type'] == parent_type:
            return n
    return None
    

### this assumes a particular hashing scheme. 
# def channel_node(G, addrhash):
#     '''
#     Return channel node associated with address hash
#     '''
#     a = str(addrhash)
#     iconn,islot,ichip,iaddr = [int(n)-1 for n in [a[0], a[1], a[2], a[3:5]]]
#     edgepath = [
#         ('wib',         'slot',         islot),
#         ('board',       'connector',    iconn),
#         ('chip',        'spot',         ichip),
#         ('channel',     'address',      iaddr)
#     ]
#     # Fixme: should not hardcode a single APA name here!
#     return child_by_path(G, 'apa', edgepath)


def flatten_to_conductor(G, channel_hash):
    '''
    Flatten the graph to the conductor level.

    The channel_hash is a callable like apa.channel_hash().
    '''
    ret = list()
    apa = 'apa'

    spots_in_layer = [40,40,48]
    boxes_in_face = 10

    for wib in neighbors_by_type(G, apa, 'wib'):
        ind_slot = G[apa][wib]['slot']

        for board in neighbors_by_type(G, wib, 'board'):
            face = parent(G, board, 'face')
            ind_connector = G[wib][board]['connector']
            ind_box = G[face][board]['spot']
            ind_side = G[apa][face]['side']

            for chip in neighbors_by_type(G, board, 'chip'):
                ind_chip = G[board][chip]['spot']

                for channel in neighbors_by_type(G, chip, 'channel'):
                    conductor = parent(G, channel, "conductor")

                    ind_address = G[chip][channel]['address']
                    ind_layer = G[board][conductor]['layer']
                    ind_spot = G[board][conductor]['spot']

                    wires = list(neighbors_by_type(G, conductor, 'wire'))
                    wires.sort(key = lambda w: G[conductor][w]['segment'])
                    nwires = len(wires)
                    wire0 = wires[0]
                    plane = parent(G, wire0, 'plane')
                    ind_wip = G[plane][wire0]['wip']
                    ind_plane = G[face][plane]['plane']
                    if ind_plane != ind_layer:
                        raise ValueError('Inconsistent graph, layer and plane differ')

                    # wire attachment number along top of APA frame.
                    ind_wan = ind_box * spots_in_layer[ind_layer] + ind_spot
                    ind_wanglobal = ind_wan + spots_in_layer[ind_layer]*boxes_in_face*ind_side

                    one = dict(channel = channel_hash(ind_connector, ind_slot, ind_chip, ind_address))
                    for k,v in locals().items():
                        if k.startswith('ind_'):
                            one[k[4:]] = v
                    ret.append(one)
    return ret

                        

def to_celltree_wires(G, channel_ident, face='face0'):
    '''
    Return list of tuples: (ch, plane, wip, sx, sy, sz, ex, ey, ez)

    corresponding to rows in the "ChannelWireGeometry" file used by

    https://github.com/bnlif/wire-cell-celltree

    for the wires in the given face.

    Note: this only returns the one face of wires but channel numbers
    are calculated with full knowledge of wrapped wires.
    '''
    ret = list()

    planes = list(neighbors_by_type(G, face, 'plane'))
    planes.sort(key = lambda p : G[face][p]['plane'])
    for plane in planes:
        wires = list(neighbors_by_type(G, plane, 'wire'))
        wires.sort(key = lambda w : G[plane][w]['wip'])
        iplane = G[face][plane]['plane']
        for wire in wires:
            iwire = G[plane][wire]['wip']

            pts = list(neighbors_by_type(G, wire, 'point'))[:2]
            head, tail = pts[0:2]
            if G[wire][head]['endpoint'] == 1:
                head, tail = pts[1], pts[0]
            ecm = [r/units.cm for r in G.node[head]['pos']]
            scm = [r/units.cm for r in G.node[tail]['pos']]

            chident = channel_ident(G, wire)
            one = [chident, iplane, iwire] + scm + ecm
            #print one
            ret.append(tuple(one))
    return ret


def to_schema(G, P, channel_ident):
    '''
    Return a wirecell.util.wires.schema store filled with information
    from connection graph G starting from given face.
    '''
    # n.b. this is called from the CLI main.

    from . import schema
    m = schema.maker()

    # fixme: currently only support one APA
    iapa = 0
    apa = P.apa

    face_indices = list()
    for face in neighbors_by_type(G, apa, 'face'):
        iface = G[apa][face]['side']

        sign = +1
        if iface == 1:          # "back" face
            sign = -1

        planes = list(neighbors_by_type(G, face, 'plane'))
        planes.sort(key = lambda p : G[face][p]['plane'])
        plane_wires = [list() for _ in planes] # temp stash
        for plane in planes:
            wires = list(neighbors_by_type(G, plane, 'wire'))
            wires.sort(key = lambda w : G[plane][w]['wip'])
            iplane = G[face][plane]['plane']
            for wire in wires:
                wip = G[plane][wire]['wip']

                pts = list(neighbors_by_type(G, wire, 'point'))[:2]
                head, tail = pts[0:2]
                if G[wire][head]['endpoint'] == 1:
                    head, tail = pts[1], pts[0]
                hpos = G.node[head]['pos']
                tpos = G.node[tail]['pos']
                h_id = m.make('point', sign*hpos.x, hpos.y, sign*hpos.z)
                t_id = m.make('point', sign*tpos.x, tpos.y, sign*tpos.z)

                conductor = parent(G, wire, 'conductor')
                segment = G[wire][conductor]['segment']
                chident = channel_ident(G, wire)
                wire_id = m.make('wire', wip, chident, segment, t_id, h_id)
                plane_wires[iplane].append(wire_id)

        wire_plane_indices = list()
        for iplane, wire_list in enumerate(plane_wires):
            # note, this assumes an APA with a "portrait" aspect ratio
            if iplane == 0:
                wire_list.sort(key = lambda w: -1*m.wire_ypos(w))
            elif iplane == 1:
                wire_list.sort(key = m.wire_ypos)
            else:
                wire_list.sort(key = lambda w: sign*m.wire_zpos(w))
            wpid = schema.wire_plane_id(iplane, iface, iapa)
            index = m.make("plane", iplane, wire_list)
            wire_plane_indices.append(index)   
        fi = m.make("face", iface, wire_plane_indices)
        face_indices.append(fi)
    m.make("anode", iapa, face_indices)
    return m.schema()
        

def wires_in_plane(G, plane):
    '''
    Return set of wire nodes connected to the given plane node.
    '''
    return neighbors_by_type(G, plane, 'wire')

def wires_in_chip(G, chip, intermediates=False):
    '''
    Return set of wire nodes connected to a chip node.  If
    intermediates is true, return the conductor and channel nodes that
    form the connection to the wires.
    '''
    channels = neighbors_by_type(G, chip, 'channel')
    conductors = set()
    for ch in channels:
        cs = neighbors_by_type(G, ch, 'conductor')
        conductors.update(cs)

    wires = set()
    for cond in conductors:
        w = neighbors_by_type(G, cond, 'wire')
        wires.update(w)

    if intermediates:
        return channels | conductors | wires
    return wires

def wires_graph(G, wires):
    '''
    Return a new graph with wire endpoints as nodes and a dictionary of 2D points
    '''
    newG = networkx.DiGraph()
    pos = dict()
    for wire in wires:
        pt1, pt2 = neighbors_by_type(G, wire, 'point')
        if G[wire][pt1]['endpoint'] == 2:
            pt1, pt2 = pt2, pt1
        pos1 = G.node[pt1]['pos']
        pos2 = G.node[pt2]['pos']
        pos[pt1] = (pos1.z, pos1.y)
        pos[pt2] = (pos2.z, pos2.y)
        newG.add_edge(pt1, pt2)
    return newG, pos

def conductors_graph(G, conductors):
    '''
    Like wires graph but swap sign of the 2D "X" (3D "Z") coordinates
    so a conductor zig-zags across a transparent frame.
    '''
    newG = networkx.DiGraph()
    pos = dict()
    for icond, cond in enumerate(conductors):
        wires = neighbors_by_type(G, cond, 'wire')
        for wire in wires:
            seg = G[cond][wire]['segment']
            sign = 1
            style="solid"
            if seg%2:
                sign = -1
                style="dashed"
            pt1, pt2 = neighbors_by_type(G, wire, 'point')        
            if G[wire][pt1]['endpoint'] == 2:
                pt1, pt2 = pt2, pt1
            pos1 = G.node[pt1]['pos']
            pos2 = G.node[pt2]['pos']
            pos[pt1] = (sign*pos1.z, pos1.y)
            pos[pt2] = (sign*pos2.z, pos2.y)
            newG.add_edge(pt1, pt2, style=style, icolor=icond)

    return newG, pos
