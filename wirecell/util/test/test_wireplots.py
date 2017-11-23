#!/usr/bin/env python
from wirecell.util.wires import apa, graph
from itertools import chain
from collections import defaultdict
import networkx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
    with PdfPages('test_plot_chip.pdf') as pdf:
        for chip in P.chip:
            print chip
            plt.title(chip)
            channels = graph.neighbors_by_type(G, chip, 'channel')
            conductors = chain.from_iterable([graph.neighbors_by_type(G, ch, 'conductor') for ch in channels])
            wg, wpos = graph.conductors_graph(G, conductors)
            networkx.draw(wg, pos=wpos, arrows=False)
            pdf.savefig()
            plt.close()

            

def test_plot_board():
    with PdfPages('test_plot_board.pdf') as pdf:
        for board in P.board:
            plt.title(board)

            bg = networkx.DiGraph()
            bpos = dict()
            chips = graph.neighbors_by_type(G, board, 'chip')
            print board, len(chips), chips
            for chip in chips:
                channels = graph.neighbors_by_type(G, chip, 'channel')
                print chip, len(channels)
                conductors = chain.from_iterable([graph.neighbors_by_type(G, ch, 'conductor') for ch in channels])
                cg, cpos = graph.conductors_graph(G, conductors)
                bg = networkx.compose(bg, cg)
                bpos.update(cpos)

            networkx.draw(bg, pos=bpos, arrows=False)
            pdf.savefig()
            plt.close()

def test_plot_wib():
    with PdfPages('test_plot_wib.pdf') as pdf:

        boards_by_connector = defaultdict(list)
        for wib in P.wib:
            boards_on_wib = graph.neighbors_by_type(G, wib, 'board')
            for b in boards_on_wib:
                iconnector = G[wib][b]['connector']
                boards_by_connector[iconnector].append((wib,b))
        print boards_by_connector
        for iconnector, wib_boards in sorted(boards_by_connector.items()):
            for wib,board in wib_boards:
                plt.title("%s connector %s %s" % (wib, iconnector, board))
                bg = networkx.DiGraph()
                bpos = dict()
                chips = graph.neighbors_by_type(G, board, 'chip')
                for chip in chips:
                    channels = graph.neighbors_by_type(G, chip, 'channel')
                    print chip, len(channels)
                    conductors = chain.from_iterable([graph.neighbors_by_type(G, ch, 'conductor') for ch in channels])
                    cg, cpos = graph.conductors_graph(G, conductors)
                    bg = networkx.compose(bg, cg)
                    bpos.update(cpos)
                networkx.draw(bg, pos=bpos, arrows=False)
                pdf.savefig()
                plt.close()
    
            
