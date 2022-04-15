#!/usr/bin/env python
'''
Support for cluster related "tap" files.

A "tap" file is a JSON such as written individually by JsonClusterTap
or into a tar stream with ClusterFileSink.
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
        dat = json.load(open(filename))
        if not isinstance(dat, list):
            dat = [dat]
        for count, one in enumerate(dat):
            yield make_nxgraph(f'{path.stem}_{count}', one)
        return

    for fname, fdata in ario.load(filename).items():
        yield make_nxgraph(fname, fdata)

