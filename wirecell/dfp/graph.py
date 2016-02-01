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


def validate(nxgraph, desc):
    '''Validate a graph against the node descriptors.  
    The <desc> is a dictionary keyed by type with values of wirecell.dfp.nodetype.NodeType.'''

    for t,h,dat in nxgraph.edges(data=True):
        tail = desc[dekey(t)[0]]
        head = desc[dekey(h)[0]]            
        tp = dat['tail_port']
        hp = dat['head_port']

        if tail.output_types[tp] != head.input_types[hp]:
            raise ValueError, 'Port data type mismatch for t1=%s[%d] and t2=%s[%d]:\nt1: %s\nt2: %s\n' % \
                (t, tp,
                 h, hp,
                 tail.output_types[tp], 
                 head.input_types[hp])

    return






