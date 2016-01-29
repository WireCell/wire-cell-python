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
        
def connect(nxgraph, k1, k2, p1, p2, **kwds):
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
            raise ValueError, 'Port data type mismatch: (%s[%d] = %s) != (%s[%d] = %s)' % \
                (t, tp, tail.output_types[tp], h, hp, head.input_types[hp])

    return

def make_dot_port_string(ishead, edges):
    if not edges:
        return ""

    letter = "o"
    port_key = "tail_port"
    if ishead:
        letter = "i"
        port_key = "head_port"

    ports = dict()
    for ind, edge in enumerate(edges):
        t,h,dat = edge
        port_num = dat[port_key]
        port_name = letter+str(port_num)
        port_label = port_name.upper()
        dt = dat.get('data_type')
        if (dt):
            port_label += "(%s)" % dt
        ports[port_name] = port_label

    items = sorted(ports.items())
    inner = '|'.join([ "<%s>%s" % p for p in items ])
    return "{|%s|}"%inner


def to_dot(nxgraph):
    '''
    Return a GraphViz dot file string. 
    '''
    ag = pgv.AGraph(directed=True, strict=False)
    ag.node_attr['shape'] = 'record'
    for nn in nxgraph.nodes():
        headstr = make_dot_port_string(1, nxgraph.in_edges(nn, data=True))
        tailstr = make_dot_port_string(0, nxgraph.out_edges(nn, data=True))
        
        lines = []
        if headstr: lines.append(headstr)
        lines.append("{%s}" % nn)
        if tailstr: lines.append(tailstr)
        
        label = '|'.join(lines)
        label = "{" + label + "}"
        ag.add_node(str(nn), label=label)
    for nt,nh,nd in nxgraph.edges(data=True):
        key = ' {tail_port}-{head_port} '.format(**nd)
        dt = nd.get('data_type')
        if dt:
            key += "(%s)" % dt
        ag.add_edge(nt,nh, key=key, label=key,
                    tailport='o'+str(nd.get('tail_port',0)),
                    headport='i'+str(nd.get('head_port',0)))

    return ag
def to_dots(nxgraph):
    return to_dot(nxgraph).string()


# def from_dots(dotstr):

#     ag = pgv.AGraph(dotstring)
#     for an in ag.nodes():
#         type = an.attr.get('type')
#         name = an.attr.get('name',None)
#         self.node(type,name);

#     for ae in ag.edges():
#         at,ah = map(str,ae)
#         self.connect(at,ah,int(ae.attr.get('tail_port','0')),int(ae.attr.get('head_port','0')))
    
