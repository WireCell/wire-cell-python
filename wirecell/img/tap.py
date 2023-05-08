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
from numpy.core.records import fromarrays
import networkx as nx

from collections import Counter

from wirecell.util import ario, jsio

import logging
log = logging.getLogger("wirecell.img")

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
        vdesc = vtx['ident']
        gr.add_node(vdesc, desc=vdesc, code=vtx['type'], **vtx['data'])
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
        log.warning(f'Warning: unexpected array: "{key}"')

    return pghd


class PgFiller:

    # cluster array schema
    dtypes = dict(
        a=[                     # 6
            ('desc',int), ('ident',int),
            ('val',float), ('unc',float),
            ('index',int), ('wpid',int)],
        w=[                     # 12
            ('desc',int),('ident',int),
            ('index',int),('seg',int),('chid',int),('wpid',int),
            ('tailx',float),('taily',float),('tailz',float),
            ('headx',float),('heady',float),('headz',float)],
        b=[                     # 39
            ('desc',int), ('ident',int),
            ('value',float), ('error',float),
            ('faceid',int),('sliceid',int), # 6
            ('start',float),('span',float),
            ('min1',int), ('max1',int),
            ('min2',int), ('max2',int),
            ('min3',int), ('max3',int), # 14
            ('ncorners',int)] + [(f'c{i}',int) for i in range(24)],
        s=[                     # 7
            ('desc',int), ('ident',int),
            ('value',float), ('error',float),
            ('frameid',int),
            ('start',float),('span',float)],
        m=[                     # 5
            ('desc',int), ('ident',int),
            ('val',float), ('unc',float),
            ('wpid',int)]
        )

    def __init__(self, gr, pg):
        self.gr = gr
        self.pg = pg
        for nc, ntype in node_types.items():
            ndat=self.get_ndat(nc)
            getattr(self, f'add_{ntype}')()
        for ec in edge_types:
            self.add_edges(ec)
        

    def get_ndat(self, nc):
        ntype = node_types[nc]
        return self.pg[ntype].x

    def get_arr(self, nc):
        ntype = node_types[nc]
        ndat = self.pg[ntype].x
        return fromarrays(ndat.T, dtype=self.dtypes[nc])

    def get_both(self, nc):
        ntype = node_types[nc]
        ndat = self.pg[ntype].x
        arr = fromarrays(ndat.T, dtype=self.dtypes[nc])
        return (ndat,arr)


    def debug(self, context):
        node_counter = Counter(dict(self.gr.nodes(data='code')).values())
        log.debug(f'{context}: {node_counter}')
    

    def add_common(self, nc, arr, **data):
        '''
        Common creation of nodes (except s/c).
        '''
        # all node types have these two
        descs = data["desc"] = arr["desc"]
        data["ident"] = arr["ident"]
        for irow, desc in enumerate(data["desc"]):
            params = dict(code=nc)
            for k,v in data.items():
                params[k] = v[irow]
            self.gr.add_node(desc, **params)
        self.debug(f'add {nc}')
        return descs

    def add_activity(self):
        '''
        Activity gets coalesced back to channels.
        '''
        # this is all DIY
        arr = self.get_arr("a")
        descs = arr["desc"]
        idents = arr["ident"]
        indexs = arr["index"]
        wpids = arr["wpid"]
        vals = arr["val"]
        uncs = arr["unc"]

        for irow, desc in enumerate(descs):
            # this re-adds same c node many times
            self.gr.add_node(desc, code="c", desc=desc,
                             val=vals[irow], unc=uncs[irow],
                             index=indexs[irow],
                             ident=idents[irow], wpid=wpids[irow])

    def add_blob(self):
        ndat, arr = self.get_both("b")
        data = {k:arr[k] for k in 'value error faceid sliceid start span'.split()}
        descs = self.add_common("b", arr, **data)
        
        corners = ndat[:,15:].reshape((-1,12,2))
        starts = data['start']
        starts = numpy.tile(starts.reshape(-1,1), 12)
        starts = starts.reshape((-1,12,1))
        corners = numpy.dstack((starts, corners))
        for irow,desc in enumerate(descs):
            self.gr.nodes[desc]["corners"] = corners[irow]
            self.gr.nodes[desc]["bounds"] = numpy.array(ndat[irow][8:14].reshape(3,2), dtype=int)

    def add_measure(self):
        arr = self.get_arr("m")
        data = {k:arr[k] for k in "val unc wpid".split()}
        self.add_common("m", arr, **data)

    def add_slice(self):
        ndat, arr = self.get_both("s")
        arr = fromarrays(ndat.T, dtype=self.dtypes["s"])
        data = {k:arr[k] for k in "frameid start span".split()}
        data['signal'] = [dict() for _ in range(arr["desc"].size)] # filled later
        self.add_common("s", arr, **data)

    def add_wire(self):
        arr = self.get_arr("w")
        data = {k:arr[k] for k in "index seg chid wpid".split()}
        descs = self.add_common("w", arr, **data)
        for irow, desc in enumerate(descs):
            for ep in ['tail','head']:
                self.gr.nodes[desc][ep]={c:arr[f'{ep}{c}'][irow] for c in "xyz"};

    def add_edges(self, ec):
        '''
        Add all edges with given two-letter edge code.

        Edges to activity (a-nodes) are changed to edges to channel (c-nodes).
        '''
        tc,hc = ec
        ekey = (node_types[tc], ec, node_types[hc])
        # these are rows
        edat = self.pg[ekey].edge_index

        nedges = self.gr.number_of_edges()
        assert(edat.shape[1] == 3)
        tarr = self.get_arr(tc)
        harr = self.get_arr(hc)
        uniq = set()
        for edesc,trow,hrow in edat:
            key = (trow,hrow)
            if key in uniq:
                continue
            uniq.add(key)
            tvtx = tarr["desc"][trow]
            hvtx = harr["desc"][hrow]

            if ec == "as":      # this is an entry in slice signal
                tdat = self.gr.nodes[tvtx]
                hdat = self.gr.nodes[hvtx]
                hdat['signal'][tdat['ident']] = dict(val=tarr["val"][trow], unc=tarr["unc"][trow])
                continue

            self.gr.add_edge(tvtx, hvtx, desc=edesc)
        nedges = self.gr.number_of_edges() - nedges


def pg2nx(name, pg):
    '''Convert a pg graph to an nx graph.

    The pg graph should be as returned by make_pggraph().

    The nx graph will be returned by make_nxgraph().

    In particular, cluster array activity nodes will be created to
    cluster graph channel nodes and slice-activity edges will be
    converted to entries in a "signal" attribute on slice nodes.

    nx node values are those of the original cluster graph vertex
    descriptor.
    '''

    # See ClusterArrays.org for column definitions
    # See ClusterHelpersJsonify for nx schema

    return PgFiller(nx.Graph(name=name), pg).gr

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
