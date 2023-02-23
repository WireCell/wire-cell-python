#!/usr/bin/env python3
'''
Dump out signatures of blobs for debugging
signature: [tmin, tmax, umin, umax, vmin, vmax, wmin, wmax]
'''
from wirecell import units
import matplotlib.pyplot as plt
import numpy as np
import math
from . import dump_blobs as db
from . import clusters
import networkx as nx
import matplotlib.pyplot as plt

TICK = 500

def test():

    def filter_graph(graph, nodes):
        subgraph = graph.subgraph(nodes)
        return subgraph

    def connected_component_subgraphs(G):
        return [G.subgraph(c) for c in nx.connected_components(G)]

    # Example usage
    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (2, 3), (4, 5), (5, 6)])
    filtered_nodes = [1, 2, 4]
    filtered_graph = filter_graph(G, filtered_nodes)

    plt.figure()
    nx.draw(G, with_labels=True)
    plt.figure()
    nx.draw(filtered_graph, with_labels=True)
    for subg in connected_component_subgraphs(G) :
        plt.figure()
        nx.draw(subg, with_labels=True)
    plt.show()


def nodes_oftype(gr, typecode):
    return [n for n,d in gr.nodes.data() if d['code'] == typecode]

def neighbors_oftype(gr, node, typecode):
    ret = list()
    for nn in gr[node]:
        if gr.nodes[nn]['code'] == typecode:
            ret.append(nn)
    return ret

def csignature(gr, bc):
    '''
    bc: one blob cluster from bgr
    '''
    bsigs = []
    for bnode in bc:
        bsig = db.bsignature(gr, bnode)
        if bsig is None:
            # print(f'bsig of {bnode} is None')
            continue
        bsig.append(gr.nodes[bnode]['ident'])
        bsigs.append(np.array(bsig))
    if len(bsigs) == 0:
        # print(f'len(bsig) for {bc} is 0')
        return None
    bsigs = np.array(bsigs)
    min_start = min(bsigs[:,0])
    max_start = max(bsigs[:,0])
    nblob = bsigs.shape[0]
    min_u = min(bsigs[:,2])
    max_u = max(bsigs[:,3])
    min_v = min(bsigs[:,4])
    max_v = max(bsigs[:,5])
    min_w = min(bsigs[:,6])
    max_w = max(bsigs[:,7])
    charge_u = sum(bsigs[:,8])
    charge_v = sum(bsigs[:,9])
    charge_w = sum(bsigs[:,10])
    min_bid = min(bsigs[:,11])
    return np.array([min_start, max_start, nblob,
                     min_u, max_u, min_v, max_v, min_w, max_w,
                     charge_u, charge_v, charge_w,
                     min_bid])

def _sort(arr):
    ind = np.lexsort((arr[:,8],arr[:,7],arr[:,6],arr[:,5],arr[:,4],arr[:,3],arr[:,2],arr[:,1],arr[:,0]))
    arr = np.array([arr[i] for i in ind])
    return arr


def dump_bb_clusters(gr):
    # graph with only blobs
    bnodes = nodes_oftype(gr,'b')
    print(f'#bnodes: {len(bnodes)}')
    bgr = gr.subgraph(bnodes)
    csigs = []
    for bc in nx.connected_components(bgr):
        csig = csignature(gr, bc)
        if csig is None:
            continue
        csigs.append(csig)
    csigs = np.array(csigs)
    csigs = _sort(csigs)
    print(f'#b in clusters: {sum(csigs[:,2])}')
    # csigs = csigs[csigs[:,0]==400,:]
    print(csigs.shape)
    for i in range(csigs.shape[0]):
        print(csigs[i,0:2], csigs[i,2],
            csigs[i,3], ',', csigs[i,4]+1, ',',
            csigs[i,5], ',', csigs[i,6]+1, ',',
            csigs[i,7], ',', csigs[i,8]+1,
            csigs[i,9:])