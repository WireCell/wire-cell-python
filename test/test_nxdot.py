import networkx as nx
g = nx.DiGraph()

g.add_node("A:a", type="A", name="a")
g.add_node("B", type="B")
g.add_edge("A:a", "B", label="0,1")

g.add_node("B:b", type="B", name="b")
g.add_edge("A:a", "B:b", label="1,1")


dot = nx.to_agraph(g);
print "g:",dot

g2 = nx.from_agraph(dot)
print "g2:",nx.to_agraph(g2);

print g2["A:a"]["B:b"]

nx.write_dot(g,"test_nxdot.dot")
