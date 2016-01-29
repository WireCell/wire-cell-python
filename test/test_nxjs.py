#!/usr/bin/env python

# http://networkx.github.io/documentation/latest/examples/javascript/force.html

import json
import networkx as nx
from networkx.readwrite import json_graph

G = nx.barbell_graph(6,3)
# this d3 example uses the name attribute for the mouse-hover value,
# so add a name to each node
for n in G:
    G.node[n]['name'] = n
# write json formatted data
d = json_graph.node_link_data(G) # node-link format to serialize
#d = json_graph.adjacency_data(G)
# write json
print json.dumps(d, indent=2)
