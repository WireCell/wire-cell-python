#!/usr/bin/env python
from wirecell import units
from wirecell.util.wires import apa, graph
from itertools import chain
from collections import defaultdict
import networkx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

desc = apa.Description();
G,P = apa.graph(desc)

xkcd_color = ['xkcd:red', 'xkcd:tangerine', 'xkcd:goldenrod',
              'xkcd:green', 'xkcd:blue', 'xkcd:purple',
              'xkcd:dark purple',
              'xkcd:violet',
              'xkcd:indigo',
              'xkcd:black',
              'xkcd:bright blue',
              'xkcd:baby blue',
              'xkcd:royal blue',
              'xkcd:periwinkle',
              'xkcd:fuchsia',
              'xkcd:aqua green',
               ]


def test_plot_plane_chip():
    inp = graph.wires_in_plane(G, P.plane[0])
    inc = graph.wires_in_chip(G, P.chip[8])
    wires = inp.intersection(inc)
    wg,wpos = graph.wires_graph(G, wires)
    networkx.draw(wg, pos=wpos, with_labels=True,width=0.1)
    plt.savefig('test_plot_plane_chip.pdf')

def test_plot_conductor():
    wg, wpos = graph.conductors_graph(G, [P.conductor[0]])
    networkx.draw(wg, pos=wpos, with_labels=True)
    plt.savefig('test_plot_conductor.pdf')

def test_plot_chip():
    with PdfPages('test_plot_chip.pdf') as pdf:
        for chip in P.chip:
            plt.title(chip)
            channels = graph.neighbors_by_type(G, chip, 'channel')
            conductors = list(chain.from_iterable([graph.neighbors_by_type(G, ch, 'conductor') for ch in channels]))
            wg, wpos = graph.conductors_graph(G, conductors)

            edges = wg.edges(data=True)
            styles = list()
            colors = list()
            for n1,n2,dat in edges:
                styles.append(dat['style'])
                icolor = dat['icolor']
                colors.append(xkcd_color[icolor%len(xkcd_color)])
            networkx.draw_networkx_edges(wg, pos=wpos, style=styles,
                                         width=0.1, arrows=False,
                                         edge_color=colors)
            print chip,len(channels),len(conductors)
            pdf.savefig()
            plt.close()
            

def test_plot_board(debug=False):
    penwidth=0.05
    if debug:
        penwidth=0.5
    with PdfPages('test_plot_board.pdf') as pdf:
        for iboard, board in enumerate(P.board):
            
            plt.title("Board %d" % (iboard+1,))

            bg = networkx.DiGraph()
            bpos = dict()
            chips = graph.neighbors_by_type(G, board, 'chip')
            edges = list()
            colors = list()
            styles = list()
            print 'board:',board, len(chips), chips
            for ichip, chip in enumerate(chips):
                channels = graph.neighbors_by_type(G, chip, 'channel')
                conductors = chain.from_iterable([graph.neighbors_by_type(G, ch, 'conductor') for ch in channels])
                cg, cpos = graph.conductors_graph(G, conductors)

                cstyles = [e[2]['style'] for e in cg.edges(data=True)]
                cedges = list(cg.edges())
                n_edges = len(cedges)
                colors += [xkcd_color[ichip]]*n_edges
                styles += cstyles
                edges += cedges

                bg = networkx.compose(bg, cg)
                bpos.update(cpos)

            networkx.draw_networkx_edges(bg, edgelist=edges,
                                         pos=bpos, style=styles,
                                         edge_color=colors,
                                         width=penwidth, arrows=False)
            assert(len(edges) == len(colors))
            if debug:
                plt.xlim(-1300,-700)
                plt.ylim(2300,3000)
                return
            plt.axes().set_aspect('equal', 'box')
            plt.xlabel('$z_c$ (mm)')
            plt.ylabel('$y_c$ (mm)')
            pdf.savefig(dpi=900)
            plt.close()
            

def test_plot_wib_wires():
    with PdfPages('test_plot_wib_wires.pdf') as pdf:

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
                networkx.draw(bg, pos=bpos, arrows=False, width=0.1)
                pdf.savefig()
                plt.close()
            
def test_plot_wib ():
    'Plot all wib connections'

    newG = networkx.Graph()
    newP = dict()

    conn_colors = ['xkcd:sky blue', 'xkcd:deep pink',
                   'xkcd:sky blue', 'xkcd:deep pink']

    def wib_pos(slot):
        return (-2 + slot - 0.5, 0)
    def conn_pos(slot, conn):
        scale=0.4
        square_pts = [(-1,-1), (1,-1), (1,1), (-1,1)]
        line_pts = [(0,-2),(0,-1),(0,1),(0,2)]
        pts = line_pts

        wp = wib_pos(slot)
        pt = pts[conn]
        return (wp[0]+scale*pt[0], wp[1]+scale*pt[1])
    def board_pos(side, spot):
        if side == 0:
            return (-5 + spot, -3)
        return (4 - spot, +3)

    apa = P.apa

    for wib in P.wib:
        nwib = "w%d"%(int(wib[3:])+1,)
        newG.add_node(nwib, color="xkcd:white", shape='o')


        islot = G[apa][wib]['slot']
        newP[nwib] = wib_pos(islot)
        boards_on_wib = graph.neighbors_by_type(G, wib, 'board')
        for board in boards_on_wib:
            face = list(graph.neighbors_by_type(G, board, 'face'))[0]
            ispot = G[face][board]['spot']

            iside = G[face][apa]['side']
            nboard = "b%d"%(int(board[5:])+1,)

            iconn = G[wib][board]['connector']
            nconn = "%d%d" % (islot+1, iconn+1)
            newG.add_node(nconn, color=conn_colors[iconn], shape='d')
            newP[nconn] = conn_pos(islot, iconn)

            newG.add_node(nboard, color=conn_colors[iconn], shape='s')
            newP[nboard] = board_pos(iside, ispot)

            newG.add_edge(nwib,nconn)
            newG.add_edge(nconn,nboard)
            

    colors = [n[1]['color'] for n in newG.nodes(data=True)]
    shapes = [n[1]['shape'] for n in newG.nodes(data=True)]

    networkx.draw(newG, pos=newP,
                  node_color=colors,
#                  node_shape=shapes,
                  with_labels=True)

    plt.savefig('test_plot_wib.pdf')
