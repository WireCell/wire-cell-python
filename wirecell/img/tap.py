#!/usr/bin/env python
'''
I/O for cluster files.

A cluster file is an archive (zip or tar, compressed or not) holding
an ICluster graph as a JSON object or as a set of Numpy arrays.  As a
special case, a bare .json file can be read for a single cluster
graph.
'''

import json
from pathlib import Path
from collections import namedtuple, defaultdict
import numpy
import networkx as nx

from wirecell.util import ario, jsio

def make_nxgraph(name, dat):
    '''
    Return networkx graph from dict base data 
    '''
    gr = nx.Graph(name=name)
    for vtx in dat['vertices']:
        gr.add_node(vtx['ident'], code=vtx['type'], **vtx['data'])
    for edge in dat['edges']:
        gr.add_edge(*edge)
    return gr

# The set of node types in an ICluster graph.
node_types = dict(c="channel", w="wire", b="blob", s="slice", m="measure")
edge_types = ('cw','bs','bw','bb','cm','bm')

def make_pggraph(name, dat):
    '''Return cluster graph data represented in a manner similar to
    torch_geometric.data.HeteroData.

    The dat dictionary provides arrays matching the WCT Cluster Arrays
    schema (see ClusterArrays.org file).  They keys of this dictionary
    are expected to be of the forms:

    <nodecode>nodes and <edgecode>edges

    Where <nodecode> is in (c,w,b,s,m) and <edgecode> are particular
    combinations of node codes in alphabetical order as allowed by the
    ICluster graph schema (see raygrid.pdf doc).

    The returned data structure matches the schema of
    pytorch-geometric's HeteroData class.

    The node type names are taken to be the full names corresponding
    to the node type codes.  For example:

    >>> pghd = make_pgraph(name, dat)

    >>> channel_features = pghd["channel"].x

    ICluster graph edges only provide connectivity information and
    thus edge types are only distinquished by the types of the edge
    endpoint nodes.  HeterData requires an edge type identifer and the
    two-letter edgecode is used.  For example,

    >>> blob_slice_edges = pghd["blob","bs","slice"].edge_index

    '''

    # The HeteroData interface we mimic
    pghd = dict()
    Node = namedtuple("Node", "x")
    Edge = namedtuple("Edge", "edge_index")

    for key, arr in dat.items():
        if key.endswith("nodes"):
            nc = key[0]
            nt = node_types[nc]
            pghd[nt] = Node(x=arr)
            continue
        if key.endswith("edges"):
            ec = key[:2]
            nt1 = node_types[ec[0]]
            nt2 = node_types[ec[1]]
            pghd[nt1,ec,nt2] = Edge(edge_index=arr)
            continue
        print(f'Warning: unexpected array: "{key}"')

    return pghd

def load_jsio(filename, path=(), **kwds):
    '''
    Load a file assuming it is a JSON-like file
    '''
    path = Path(filename)
    got = jsio.load(filename, path, **kwds)
    if not isinstance(got, list):
        got = [list]
    for count, one in enumerate(got):
        yield make_nxgraph(f'{path.stem}_{count}', one)

def group_keys(arf):
    '''
    Group sequence of keys into single .json or list of .npy keys
    '''
    keys = list(arf.keys())

    ret = list()
    while keys:
        key = keys[0]
        member = arf.member_names[key]
        if '.json' in member:
            ret.append(key)
            keys.pop(0)
            continue

        cl,n,kind = key.split("_", 2)
        if cl != "cluster":
            raise ValueError("failed to parse")
        pre = f'cluster_{n}_'
        sub = [key]
        keys.pop(0)
        while keys and keys[0].startswith(pre):
            sub.append(keys.pop(0))
        ret.append(sub)
    return ret
        

def load_ario(filename, **kwds):
    '''
    Load a file assuming it is an archive of files.
    '''
    path = Path(filename)
    arf = ario.load(filename, False)
    
    for key in group_keys(arf):
        if isinstance(key, str): # jsio
            dat = arf[key]
            if not isinstance(dat, list):
                dat = [dat]
            for count, one in enumerate(dat):
                yield make_nxgraph(f'{path.stem}_{key}_{count}', one)
        else:
            dat = {k.split("_",2)[-1]:arf[k] for k in key}
            cl,n,kind = key[0].split("_",2)
            yield make_pggraph(f'{path.stem}_{n}', dat)

def load(filename, **kwds):
    '''
    Yield a sequence of graphs loaded from file like object.
    '''
    if filename.endswith((".json", ".json.gz", ".json.bz2", '.jsonnet')):
        for one in load_jsio(filename, **kwds):
            yield one
    else:
        for one in load_ario(filename, **kwds):
            yield one
