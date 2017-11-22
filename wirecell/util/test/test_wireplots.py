#!/usr/bin/env python
from wirecell.util.wires import apa, graph

import networkx
import matplotlib.pyplot as plt

desc = apa.Description();
G,P = apa.graph(desc)


def test_plot_plane_chip():
    inp = graph.wires_in_plane(G, P.plane[0])
    inc = graph.wires_in_chip(G, P.chip[0])
    wg,wpos = graph.wires_graph(G, wires)
    networkx.draw(wg, pos=wpos, with_labels=True)
    plt.savefig('test_plot_plane_chip.pdf')

def test_plot_conductor():
    wg, wpos = graph.conductors_graph(G, [P.conductor[0]])
    networkx.draw(wg, pos=wpos, with_labels=True)
    plt.savefig('test_plot_conductor.pdf')

def test_plot_chip():
    conductors = graph.neighbors_by_type(G, P.chip[0], 'conductor', 2)
    assert conductors
    wg, wpos = graph.conductors_graph(G, conductors)
    networkx.draw(wg, pos=wpos, with_labels=True)
    plt.savefig('test_plot_chip.pdf')
