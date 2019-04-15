#!/usr/bin/env python
'''
Load in a graph from a JsonClusterTap JSON file
'''

import json

def load(filename):
    dat = json.load(open(filename))
    import networkx as nx
    gr = nx.Graph(filename=filename)
    for vtx in dat['vertices']:
        gr.add_node(vtx['ident'], code=vtx['type'], **vtx['data'])
    for edge in dat['edges']:
        gr.add_edge(*edge)
    return gr

