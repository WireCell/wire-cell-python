import networkx as nx
g = nx.DiGraph()

g.add_node("A:a", type="A", name="a")
g.add_node("B", type="B")
g.add_edge("A:a", "B", label="0,1")

g.add_node("B:b", type="B", name="b")
g.add_edge("A:a", "B:b", label="1,1")


dot = nx.nx_agraph.to_agraph(g)
print ("g:",dot)

g2 = nx.nx_agraph.from_agraph(dot)
print ("g2:",nx.nx_agraph.to_agraph(g2))

print (g2["A:a"]["B:b"])

nx.drawing.nx_agraph.write_dot(g,"test_nxdot.dot")
