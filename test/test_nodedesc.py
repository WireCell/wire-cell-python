from wirecell.dfp import nodetype, dot
import pygraphviz as pgv

def test_nodedesc():
    'Make a wirecell.dfp.graph.Graph'
    
    addtypes = True
    rankdir='TB'
    if addtypes:
        rankdir='LR'

    ag = pgv.AGraph(name="nodetypes", directed=True, strict=False, rankdir=rankdir)
    ag.node_attr['shape'] = 'record'

    j = open("nodedesc.json").read()
    for typ,dat in nodetype.loads(j).items():
        ag.add_node(typ, label = dot.nodetype_label(dat, addtypes))

    print ag.string()

    
if '__main__' == __name__:
    test_nodedesc()
