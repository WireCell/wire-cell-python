from wirecell.dfp import graph, nodetype, dot


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
    graph.connect(g, "Digitizer", "ChannelCellSelector")
    graph.connect(g, "ChannelCellSelector", "CellSliceSink")

    desc = nodetype.loads(open("nodedesc.json").read())
    graph.validate(g, desc)

    ag = dot.gvgraph(g);
    print ag.string()

if '__main__' == __name__:
    test_gen_sim_dfp()
    

