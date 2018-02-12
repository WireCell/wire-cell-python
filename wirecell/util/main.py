import sys
import click
import numpy
from wirecell import units

@click.group("util")
@click.pass_context
def cli(ctx):
    '''
    Wire Cell Toolkit Utility Commands
    '''

@cli.command("convert-oneside-wires")
@click.argument("input-file")
@click.argument("output-file")
@click.pass_context
def convert_oneside_wires(ctx, input_file, output_file):
    '''
    Convert a a "onesided" wires description file into one suitable for WCT.

    An example file is:

    https://github.com/BNLIF/wire-cell-celltree/blob/master/geometry/ChannelWireGeometry_v2.txt

    It has columns like:
    # channel plane wire sx sy sz ex ey ez

    The output file is JSON and if it has .gz or .bz2 it will be compressed.
    '''
    from wirecell.util.wires import onesided, persist
    store = onesided.load(input_file)
    persist.dump(output_file, store)


@cli.command("convert-multitpc-wires")
@click.argument("input-file")
@click.argument("output-file")
@click.pass_context
def convert_multitpc_wires(ctx, input_file, output_file):
    '''
    Convert a "multitpc" wire description file into one suitable for
    WCT.

    Here "TPC" refers to on anode face.  That is, one APA is made up
    of two TPCs.  An example file is protodune-wires-larsoft-v1.txt
    from wire-cell-data.  It has columns like:

        # chan tpc plane wire sx sy sz ex ey ez

    And, further the order of rows of identical channel number express
    progressively higher segment count.
    '''
    from wirecell.util.wires import multitpc, persist
    store = multitpc.load(input_file)
    persist.dump(output_file, store)



@cli.command("convert-uboone-wire-regions")
@click.argument("wire-json-file")
@click.argument("csvfile")
@click.argument("region-json-file")
@click.pass_context
def convert_uboon_wire_regions(ctx, wire_json_file, csvfile, region_json_file):
    '''
    Convert CSV file to WCT format for wire regions.  Example is one
    as saved from MicroBooNE_ShortedWireList.xlsx.
    '''
    import wirecell.util.wires.persist as wpersist
    import wirecell.util.wires.regions as reg
    store = wpersist.load(wire_json_file)
    ubs = reg.uboone_shorted(store, csvfile)
    wpersist.dump(region_json_file, ubs)
    

@cli.command("plot-wires")
@click.argument("json-file")
@click.argument("pdf-file")
@click.pass_context
def plot_wires(ctx, json_file, pdf_file):
    '''
    Plot wires from a WCT JSON(.bz2) wire file
    '''
    import wirecell.util.wires.persist as wpersist
    import wirecell.util.wires.plot as wplot
    wires = wpersist.load(json_file)
    wplot.allplanes(wires, pdf_file)

@cli.command("plot-select-channels")
@click.argument("json-file")
@click.argument("pdf-file")
@click.argument("channels", nargs=-1, type=int)
@click.pass_context
def plot_select_channels(ctx, json_file, pdf_file, channels):
    '''
    Plot wires for select channels from a WCT JSON(.bz2) wire file
    '''
    import wirecell.util.wires.persist as wpersist
    import wirecell.util.wires.plot as wplot
    wires = wpersist.load(json_file)
    wplot.select_channels(wires, pdf_file, channels)


@cli.command("gen-plot-wires")
@click.argument("output-file")
@click.pass_context
def gen_plot_wires(ctx, output_file):
    '''
    Generate wires and plot them.
    '''
    import wirecell.util.wires.plot as wplot
    import wirecell.util.wires.generator as wgen
    s = wgen.onesided_wrapped()
    fig,ax = wplot.oneplane(s, 0)
    #fig,ax = wplot.allplanes(s)
    #fig,ax = wplot.allwires(s)
    fig.savefig(output_file)

@cli.command("make-wires")
@click.option('-d','--detector',
#              type=click.Choice("microboone protodune dune apa".split()),
              type=click.Choice(['apa']),
              help="Set the target detector")
# fixme: give interface to tweak individual parameters
# fixme: give way to set a graph transformation function
# fixme: give way to set a template for file generation
@click.argument("output-file")
@click.pass_context
def make_wires(ctx, detector, output_file):
    '''
    Generate a WCT "wires" file giving geometry and connectivity of
    conductor wire segments and channel identifiers.
    '''
    if detector == "apa":
        from wirecell.util.wires import apa, graph, persist
        desc = apa.Description();
        G,P = apa.graph(desc)

        store = graph.to_schema(G, apa.channel_ident)
        persist.dump(output_file, store)
        return
    click.echo('Unknown detector type: "%s"' % detector)
    sys.exit(1)

@cli.command("make-map")
@click.option('-d','--detector',
#              type=click.Choice("microboone protodune dune apa".split()),
              type=click.Choice(['apa']),
              help="Set the target detector")
@click.argument("output-file")
@click.pass_context
def make_map(ctx, detector, output_file):
    '''
    Generate a WCT channel map file giving numpy arrays.
    '''
    schema = output_file[output_file.rfind(".")+1:]
    click.echo('writing schema: "%s"' % schema)

    if detector == "apa":
        from wirecell.util.wires import apa
        if schema == "npz":
            click.echo('generating Numpy file "%s"' % output_file)
            numpy.savez(output_file, **dict(
                chip_channel_spot = apa.chip_channel_spot,
                chip_channel_layer = apa.chip_channel_layer,
                connector_slot_board = numpy.asarray(range(10),dtype=numpy.int32).reshape(2,5)+1,
                face_board_femb = numpy.asarray(range(20),dtype=numpy.int32).reshape(2,10)+1))
            return
        if schema == "tex":     # fixme: this should be moved out of here
            click.echo('generating LaTeX fragment file "%s"' % output_file)
            with open(output_file,"w") as fp:
                color = dict(u="red", v="blue", w="black")
                lines = list()
                mat = numpy.asarray([r"\textcolor{%s}{%s%02d}" % (color[p], p, w) \
                                     for p,w in apa.chip_channel_layer_spot_matrix.reshape(8*16,2)])\
                           .reshape(8,16)
                for chn, ch in enumerate(mat.T):
                    cells = ["ch%02d" % chn]
                    for chip in ch:
                        cells.append(chip)
                    lines.append(" & " .join(cells))
                end = r"\\" + "\n"
                body = end.join(lines)
                top = "&".join(["ASIC:"] + [str(n+1) for n in range(8)]) + r"\\"
                form = "r|rrrrrrrr"
                tabular = [r"\begin{center}", r"\begin{tabular}{%s}"%form, r"\hline", top,
                           r"\hline", body+r"\\", r"\hline", r"\end{tabular}",r"\end{center}",""]
                fp.write("\n".join(tabular))

                layers = dict(u=[""]*40, v=[""]*40, w=[""]*48)
                for chipn, chip in enumerate(apa.chip_channel_layer_spot_matrix):
                    for chn, (plane,wire) in enumerate(chip):
                        layers[plane][wire-1] = (chipn,chn)

                lines = list()
                for letter, layer in sorted(layers.items()):
                    nchans = len(layer)
                    nhalf = nchans // 2
                    form = "|" + "C{3.5mm}|"*nhalf
                    lines += ["",
                              #r"\tiny",
                              r"\begin{center}",
                              r"\begin{tabular}{%s}"%form,
                    ]

                    lines += [#r"\hline",
                              r"\multicolumn{%d}{c}{%s layer, first half: conductor / chip / chan} \\" % (nhalf, letter.upper()),
                              r"\hline"]
                    wires = "&".join(["%2s"%ww for ww in range(1,nhalf+1)]) + r"\\";
                    chips = "&".join(["%2s"%(cc[0]+1,) for cc in layer[:nhalf]]) + r"\\";
                    chans = "&".join(["%2s"%cc[1] for cc in layer[:nhalf]]) + r"\\";
                    lines += [wires, r"\hline", chips, chans];

                    lines += [r"\hline",
                              r"\multicolumn{%d}{c}{%s layer, second half: conductor / chip / chan} \\" % (nhalf, letter.upper()),
                              r"\hline"]
                    wires = "&".join(["%2s"%ww for ww in range(nhalf+1,nchans+1)]) + r"\\";
                    chips = "&".join(["%2s"%(cc[0]+1,) for cc in layer[nhalf:]]) + r"\\";
                    chans = "&".join(["%2s"%cc[1] for cc in layer[nhalf:]]) + r"\\";
                    lines += [wires, r"\hline", chips, chans];

                    lines += [r"\hline", r"\end{tabular}"]
                    lines += [r"\end{center}"]
                fp.write("\n".join(lines))
            return

        return


    click.echo('Unknown detector type: "%s"' % detector)
    sys.exit(1)


@cli.command("gravio")
@click.argument("dotfile")
@click.pass_context
def gravio(ctx, dotfile):
    '''
    Make a dot file using gravio of the connectivity.
    '''
    try:
        from gravio import gen, dotify
    except ImportError:
        click.echo('You need to install the "gravio" package')
        click.echo('See https://github.com/brettviren/gravio')
        sys.exit(1)
    from wirecell.util.wires import apa, graph
    desc = apa.Description();
    G,P = apa.graph(desc)

    # ['wire', 'wib', 'conductor', 'point', 'chip', 'face', 'plane', 'board', 'detector', 'apa', 'channel']
    node_colors = dict(wib='orange', chip='red', face='blue', plane='purple', board='green', channel='pink')
    def skip_node(n):
        skip_types = ['point','wire','conductor']
        nt = G.nodes[n]['type']
        return nt in skip_types

    gr = gen.Graph("dune", "graph")
    gr.node(shape='point')

    for name, params in G.nodes.items():
        if skip_node(name):
            continue
        nt = G.nodes[name]['type']        
        gr.node(name, shape='point', color=node_colors.get(nt, 'black'))

    #link_types = ['slot', 'submodule', 'pt', 'trace', 'wip', 'spot', 'side',
    #              'plane', 'place', 'address', 'segment', 'cable', 'channel']
    link_colors = dict(trace='pink', spot='red', side='blue', plane='purple', address='yellow', cable='brown', chanel='orange')
    seen_edges = set()
    for (n1,n2), params in G.edges.items():
        if skip_node(n1) or skip_node(n2):
            continue
        if (n1,n2) in seen_edges or (n2,n1) in seen_edges:
            continue
        seen_edges.add((n1,n2))
        link = params['link']
        gr.edge(n1,n2, color=link_colors.get(link, 'black'))


    d = dotify.Dotify(gr)
    dottext = str(d)
    open(dotfile,'w').write(dottext)

@cli.command("make-wires-onesided")
@click.argument("output-file")
@click.pass_context
def make_wires_onesided(ctx, output_file):
    '''
    Generate a WCT wires file. 
    '''
    import wirecell.util.wires.generator as wgen
    import wirecell.util.wires.persist as wpersist
    s = wgen.onesided_wrapped()           # fixme, expose different algs to CLI
    wpersist.dump(output_file, s)
    
@cli.command("wire-channel-map")
@click.argument("input-file")
@click.pass_context
def wire_channel_map(ctx, input_file):
    '''
    Generate a WCT channel map wires file.
    '''
    from collections import defaultdict
    import wirecell.util.wires.persist as wpersist
    s = wpersist.load(input_file)

    channel_map = defaultdict(list)

    for anode in s.anodes:
        for iface in anode.faces:
            face = s.faces[iface]
            for iplane in face.planes:
                plane = s.planes[iplane]
                for iwire in plane.wires:
                    wire = s.wires[iwire]
                    # fixme: why isn't wire.ident changing?
                    channel_map[(plane.ident,wire.channel)].append(iwire)
    
    for c,wires in sorted(channel_map.items()):
        if c[1] in range(4210, 4235):
            wires.sort()
            click.echo(c,"\t",wires)


def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
    
