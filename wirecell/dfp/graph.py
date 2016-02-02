#!/usr/bin/env python
'''
Wire Cell Data Flow Programming Graph
'''

import json
from collections import namedtuple
import networkx as nx
import pygraphviz as pgv

Graph = nx.MultiDiGraph

def key(type, name=None):
    '''
    Return a graph key for a node's type and name.
    '''
    key = type
    if name:
        key += ":" + name
    return key

def dekey(key):
    '''
    Return (type,name) tuple from key
    '''
    if not ':' in key:
        key += ':'
    return tuple(key.split(':',1))
        
def connect(nxgraph, k1, k2, p1=0, p2=0, **kwds):
    '''
    Connect node at key k1, port p1 to node k2, port p2
    '''
    nxgraph.add_edge(k1, k2, tail_port=p1, head_port=p2, **kwds)


def port_compare(what, got, want):
    if got.difference(want):
        raise ValueError, "Got more ports for %s got:%d want:%d" % (what, len(got), len(want))
    if want.difference(got):
        raise ValueError, "Want more ports for %s got:%d want:%d" % (what, len(got), len(want))

    return

def validate(nxgraph, desc):
    '''Validate a graph against the node descriptors.  
    The <desc> is a dictionary keyed by type with values of wirecell.dfp.nodetype.NodeType.'''

    for nn in nxgraph.nodes():
        nd = desc[dekey(nn)[0]]

        port_compare('input to %s' % nn,
                     set([d['head_port'] for t,h,d in nxgraph.in_edges(nn, data=True)]),
                     set(range(len(nd.input_types))))

        port_compare('output from %s' % nn,
                     set([d['tail_port'] for t,h,d in nxgraph.out_edges(nn, data=True)]),
                     set(range(len(nd.output_types))))

    for t,h,dat in nxgraph.edges(data=True):
        tail = desc[dekey(t)[0]]
        head = desc[dekey(h)[0]]            
        tp = dat['tail_port']
        hp = dat['head_port']

        # make sure connection types match
        if tail.output_types[tp] != head.input_types[hp]:
            raise ValueError, 'Port data type mismatch for t1=%s[%d] and t2=%s[%d]:\nt1: %s\nt2: %s\n' % \
                (t, tp,
                 h, hp,
                 tail.output_types[tp], 
                 head.input_types[hp])

    return



def wirecell_graph(nxgraph):
    '''Return nxgraph as a JSON string suitable for inclusion into a
    wire-cell configuration file for giving to a WireCell::DfpGraph.
    '''
    ret = list()
    for t,h,dat in nxgraph.edges(data=True):
        ttype, tname = dekey(t)
        htype, hname = dekey(h)
        d = dict(
            tail = dict(type=ttype, name=tname, port=dat['tail_port']),
            head = dict(type=htype, name=hname, port=dat['head_port']))
        ret.append(d)
    return ret
    

