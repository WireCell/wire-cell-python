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
import networkx as nx
from wirecell.util import ario

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

def load(filename):
    '''
    Yield a sequence of graphs loaded from file like object.
    '''
    path = Path(filename)
    if path.suffix in (".json",):
        dat = ario.transform(filename, open(filename).read())
        if not isinstance(dat, list):
            dat = [dat]
        for count, one in enumerate(dat):
            yield make_nxgraph(f'{path.stem}_{count}', one)
        return

    arf = ario.load(filename, False)
    for key in arf:
        member = arf.member_names[key]
        if '.json' in member:
            dat = arf[key]
            if not isinstance(dat, list):
                dat = [dat]
            for count, one in enumerate(dat):
                yield make_nxgraph(f'{path.stem}_{count}', one)
        else:
            raise ValueError("Cluster graphs in Numpy format not yet supported")

