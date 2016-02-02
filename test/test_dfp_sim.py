from wirecell.dfp import graph, nodetype, dot
from subprocess import check_output
import json

def test_gen_sim_dfp():
    g = graph.Graph()
    for ind,letter in enumerate("UVW"):
        drifter = graph.key("Drifter","Drifter"+letter)
        diffuser = graph.key("Diffuser","Diffuser"+letter)
        ductor = graph.key("PlaneDuctor","PlaneDuctor"+letter)
        graph.connect(g, "TrackDepos", drifter)
        graph.connect(g, drifter, diffuser)
        graph.connect(g, diffuser, ductor)
        graph.connect(g, ductor, "PlaneSliceMerger", 0, ind)
    graph.connect(g, "PlaneSliceMerger", "Digitizer")
    graph.connect(g, "WireSource", "Digitizer",0,1)
    graph.connect(g, "Digitizer", "ChannelCellSelector")
    graph.connect(g, "WireSource", "BoundCells")
    graph.connect(g, "BoundCells", "ChannelCellSelector",0,1)
    graph.connect(g, "ChannelCellSelector", "CellSliceSink")
    desc = nodetype.loads(open("nodedesc.json").read())
    graph.validate(g, desc)

    ag = dot.gvgraph(g);
    open("test_dfp_sim.dot","w").write(ag.string())
    print check_output("dot -Tpdf -otest_dfp_sim.pdf test_dfp_sim.dot", shell=True)

    wcg = graph.wirecell_graph(g)
    cfg = [dict(type= "TbbFlow", data=dict(dfp = "TbbDataFlowGraph", graph = wcg)),]

    json.dump(cfg, open("test_dfp_sim.cfg","w"), indent=2)

if '__main__' == __name__:
    test_gen_sim_dfp()
    

