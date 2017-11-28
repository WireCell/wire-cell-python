#!/usr/bin/env python
'''
Do fun stuff to a connectivity graph
'''

import networkx


def neighbors_by_type(G, seed, typename, radius=1):
    '''
    Return a set of all neighbor nodes withing given radius of seed
    which are of the given typename.
    '''
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
    for cond in conductors:
        wires = neighbors_by_type(G, cond, 'wire')
        for wire in wires:
            seg = G[cond][wire]['segment']
            sign = 1
            if seg%2:
                sign = -1
            pt1, pt2 = neighbors_by_type(G, wire, 'point')        
            if G[wire][pt1]['endpoint'] == 2:
                pt1, pt2 = pt2, pt1
            pos1 = G.node[pt1]['pos']
            pos2 = G.node[pt2]['pos']
            pos[pt1] = (sign*pos1.z, pos1.y)
            pos[pt2] = (sign*pos2.z, pos2.y)
            newG.add_edge(pt1, pt2)
    return newG, pos
