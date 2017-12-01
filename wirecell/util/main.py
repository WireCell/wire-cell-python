import click
import sys
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
    print wires
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
        store = graph.to_schema(G)
        persist.dump(output_file, store)
        return
    click.echo('Unknown detector type: "%s"' % detector)
    sys.exit(1)

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
            print c,"\t",wires


def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
    
