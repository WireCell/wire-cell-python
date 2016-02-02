import pygraphviz as pgv

def dotencode(string):
    return string.replace("<","&lt;").replace(">","&gt;")

def node_port_string(letter, types, addtypes=False):
    if not types:
        return None
    ports = dict()
    for ind,typ in enumerate(types):
        port_name = letter+str(ind)
        port_label = port_name.upper()
        ## C++ templated types tend to screw up dot
        if addtypes:
            port_label += "(%s)" % dotencode(typ) 
        ports[port_name] = port_label
    items = sorted(ports.items())
    inner = '|'.join([ "<%s>%s" % p for p in items ])
    return "{|%s|}"%inner



def nodetype_label(nt, addtypes=False):
    '''
    Return a GraphViz node label defined based on given NodeType <nt>.
    '''
    lines = list()
    lines.append(node_port_string("i", nt.input_types, addtypes))
    lines.append("{%s (cat:%d con:%d)}" % (nt.type, nt.category, nt.concurrency))
    lines.append(node_port_string("o", nt.output_types, addtypes))
    label = '|'.join([l for l in lines if l])
    return "{" + label + "}"



def gvgraph_nodetypes(nxgraph, nodetypes):
    '''Return a GraphViz graph made from the NX graph.  The <nodetypes> is
    a NodeType dictionary and will be used to define the nodes.
    '''
    ag = pgv.AGraph(directed=True, strict=False)
    ag.node_attr['shape'] = 'record'

    for nn in nxgraph.nodes():
        typ = str(nn)
        if ':' in typ:
            typ = typ.split(':',1)[0]
        nt = nodetypes[typ]
        ag.add_node(nn, label = nodetype_label(nt))

    for nt,nh,nd in nxgraph.edges(data=True):
        key = ' {tail_port}-{head_port} '.format(**nd)
        dt = nd.get('data_type')
        if dt:
            key += "(%s)" % dt
        ag.add_edge(nt,nh, key=key, label=key,
                    tailport='o'+str(nd.get('tail_port',0)),
                    headport='i'+str(nd.get('head_port',0)))

    return ag

    

def edge_port_string(letter, edges):
    port_key = "tail_port"
    if letter == "i":
        port_key = "head_port"

    ports = dict()
    for t,h,dat in edges:
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

def edgetype_label(nxnode, inedges, outedges):
    '''
    Return a GraphViz node label defined based on its collection of input and output edges.
    '''
    lines = list()
    lines.append(edge_port_string('i',inedges))
    lines.append(nxnode)
    lines.append(edge_port_string('o',outedges))
    label = '|'.join([l for l in lines if l])
    return label


def gvgraph(nxgraph):
    '''Return a GraphViz graph made from the NX graph.'''
    
    ag = pgv.AGraph(directed=True, strict=False, overlap='false', splines='true')
    ag.node_attr['shape'] = 'record'

    for nn in nxgraph.nodes():
        nodestring = edgetype_label(nn, nxgraph.in_edges(nn,data=True), nxgraph.out_edges(nn,data=True))
        label = "{" + nodestring + "}"
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
    
