#!/usr/bin/env python
'''
I/O for cluster files.

A cluster file is an archive (zip or tar, compressed or not) holding
an ICluster graph as a JSON object or as a set of Numpy arrays.  As a
special case, a bare .json file can be read for a single cluster
graph.

'''
import sys
import json
from pathlib import Path
from collections import namedtuple, defaultdict
import numpy
import networkx as nx

from wirecell.util import ario, jsio

def slice_channels(gr, snode):
    '''
    Return list of channel nodes reachable from slice node.
    '''
    cnodes = set()
    bnodes = [n for n in gr.neighbors(snode) if gr.nodes[n]['code'] == 'b']
    for bnode in bnodes:
        mnodes = [n for n in gr.neighbors(bnode) if gr.nodes[n]['code'] == 'm']
        for mnode in mnodes:
            cnodes.update([n for n in gr.neighbors(mnode) if gr.nodes[n]['code'] == 'c'])
    return cnodes

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

# Expand possible node array codes
node_types = dict(a="activity", w="wire", b="blob", s="slice", m="measure")
edge_types = ('aw','bs','bw','bb','am','bm','as')


def make_pggraph(name, dat):
    '''Return graph data represented in a manner similar to
      torch_geometric.data.HeteroData.

      Note: see pg2nx() to convert the graph returned by this method
      to one that is equivalent to the graph returned by
      make_nxgraph().

      The provided "dat" argument must be a dictionary providing
      arrays matching the WCT Cluster Arrays schema (see
      ClusterArrays.org file) such as read directly from a cluster
      array archive file.  They keys of this dictionary are expected
      to be of the forms:

      <nodecode>nodes and <edgecode>edges

      Where <nodecode> is a letter representing a node type and
      <edgecode> is a pair of node codes in alphabetical order

      The returned data structure matches the schema of
      pytorch-geometric's HeteroData class.

      The node type names are taken to be the full names corresponding
      to the node type codes.  For example:

      >>> pghd = make_pggraph(name, dat)

      >>> channel_features = pghd["channel"].x

      ICluster graph edges only provide connectivity information and
      thus edge types are only distinquished by the types of the edge
      endpoint nodes.  HeterData requires an edge type identifer and
      the two-letter edgecode is used.  For example,

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


def pg2nx(name, pg):
    '''Convert a pg graph such as returned by make_pggraph() to a
    networkx graph such as returned by make_nxgraph().

    This will convert activity nodes to channel nodes and
    slice-activity edges to entries in a "signal" attribute on slice
    nodes.

    '''
    gr = nx.Graph(name=name)

    # See ClusterArrays.org for column definitions
    # See ClusterHelpersJsonify for nx schema

    # nodes
    count = 0;
    row2node = dict()           # by code+row
    ident2node = dict()         # by code+ident

    for nc, ntype in node_types.items():

        def add_node(irow, ident, **kwds):
            nonlocal count
            nonlocal row2node
            nonlocal ident2node
            ident = int(ident)
            code = nc

            if code == 'a':   # transform activity row to channel node
                code = 'c'
                cid = f'c{ident}'
                if cid in ident2node: # reuse first seen
                    rid = f'c{irow}'  # alias this row
                    row2node[rid] = ident2node[cid] 
                    return

            gr.add_node(count, ident=ident, code=code, **kwds)
            row2node[f'{code}{irow}'] = count;
            ident2node[f'{code}{ident}'] = count
            count += 1

        ndat = pg[ntype].x

        # print(f'{ntype}: {ndat.shape}')

        if ntype == "activity": # this one is special
            for irow, row in enumerate(ndat):
                add_node(irow, row[0], # makes no sense: value=row[1], error=row[2],
                         index=int(row[3]), wpid=int(row[4]));
                
        if ntype == "wire":
            for irow, row in enumerate(ndat):
                add_node(irow, row[0], index=int(row[1]), seg=int(row[2]),
                         chid=int(row[3]), wpid=int(row[4]),
                         tail=dict(x=row[5], y=row[6], z=row[7]),
                         head=dict(x=row[8], y=row[9], z=row[10]))

        if ntype == "blob":
            for irow, row in enumerate(ndat):
                ncorners_index = 13
                ncorners = int(row[ncorners_index])
                corners = list()
                t0 = row[5]     # start time
                for icorn in range(ncorners):
                    cy = row[ncorners_index+1+2*icorn]
                    cz = row[ncorners_index+1+2*icorn+1]
                    corners.append((t0, cy, cz))
                add_node(irow, row[0], value=row[1], error=row[2],
                         faceid=int(row[3]), sliceid=int(row[4]),
                         start=row[5], span=row[6],
                         bounds=numpy.array(row[7:13].reshape(3,2), dtype=int),
                         corners=numpy.array(corners, dtype=int))

        if ntype == "slice":
            for irow, row in enumerate(ndat):
                # Activity gets filled below
                add_node(irow, row[0], frameid=int(row[3]),
                         start=row[4], span=row[5], signal=dict())

        if ntype == "measure":
            for irow, row in enumerate(ndat):
                add_node(irow, row[0], val=row[1], unc=row[2], wpid=int(row[3]))

    for ec in edge_types:
        tc,hc = ec
        ekey = (node_types[tc], ec, node_types[hc])
        edat = pg[ekey].edge_index

        # print(f'{ekey}: {edat.shape}')
        if tc != 'a':           # relies on 'a' l.t. all other codes
            for row in edat:
                ti,hi = row
                tn = row2node[f'{tc}{ti}']
                hn = row2node[f'{hc}{hi}']
                gr.add_edge(tn, hn)
            continue

        # special case transformation from activity to channel

        if hc == 's':           # fill slice's "activity map"
            for ti, hi in edat:
                arow = pg['activity'].x[ti]
                srow = pg['slice'].x[hi]
                snode = row2node[f's{hi}']
                sdata = gr.nodes[snode]
                sdata['signal'][int(arow[0])] = dict(val=arow[1], unc=arow[2])
                # no edge created
            continue

        if hc == 'w' or hc == 'm':
            for ti, hi in edat:
                arow = pg['activity'].x[ti]
                cident = int(arow[0])
                hn = row2node[f'{hc}{hi}']
                tn = ident2node[f'c{cident}']
                gr.add_edge(tn, hn)
            continue

    return gr


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
            raise ValueError(f"failed to parse key: {key}")
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
            name = f'{path.stem}_{n}'
            yield pg2nx(name, make_pggraph(name, dat))

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
