from wirecell.dfp import graph, nodetype, dot

# $ wire-cell -p WireCellGen -p WireCellApps -a NodeDumper
node_desc_json = '''
[
	{
		"category" : 1,
		"concurrency" : 1,
		"output_types" : [ "int", "float", "double", "int" ],
		"type" : "A"
	},
	{
		"category" : 3,
		"concurrency" : 0,
		"input_types" : ["int","float","double"],
		"type" : "B"
	}
]
'''


def test_make():
    'Make a wirecell.dfp.graph.Graph'
    g = graph.Graph()
    a = graph.key("A","a")
    b = graph.key("B","b")
    graph.connect(g, a,b, 0,0)
    graph.connect(g, a,b, 1,1)
    graph.connect(g, a,b, 2,2)
    graph.connect(g, a,b, 3,0)

    desc = nodetype.loads(node_desc_json)
    graph.validate(g, desc)

    gvgraph = dot.gvgraph_nodetypes(g, desc)
    print gvgraph.string()

    # dot = g.dumps_dot()
    # print dot
    # open('foo.dot','w').write(dot)

    # g2 = Graph()
    # g2.loads_dot(dot)
    # print g2.dumps_dot()
    
if '__main__' == __name__:
    test_make()
    

