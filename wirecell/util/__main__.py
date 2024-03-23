#!/usr/bin/env python3

import os
import sys
import math
import json
import click
import numpy
from collections import defaultdict
from wirecell import units
from wirecell.util.functions import unitify, unitify_parse
from wirecell.util.cli import context, log, jsonnet_loader
from wirecell.util.fileio import wirecell_path
from wirecell.util import jsio, detectors

@context("util")
def cli(ctx):
    '''
    Wire Cell Toolkit Utility Commands
    '''
    pass


@cli.command("convdown")
@click.option("-n", "--sampling-number", default=None, type=int,
              help="Original number of samples.")
@click.option("-t", "--sampling-period", default=None, type=str,
              help="Original sample period,eg '100*ns'")
@click.option("-N", "--resampling-number", default=None, type=int,
              help="Resampled number of samples.")
@click.option("-T", "--resampling-period", default=None, type=str,
              help="Resampled sample period, eg '64*ns'")
@click.option("-e", "--error", default=1e-6,
              help="Precision by which integer and "
              "rationality conditions are judged")
def cmd_convdown(sampling_number, sampling_period, resampling_period, resampling_number, error):
    '''
    Calculate numbers for "simultaneous convolution and downsample" method.

    Eg:

        $ wirecell-util convdown -n 625 -t '100*ns' -N 200 -T "500*ns"
        (Ta=100.0 ns, Na=625) -> 1625, (Tb=500.0 ns, Nb=200) -> 325

    '''
    from wirecell.util import lmn
    Ta = unitify(sampling_period)
    Na = sampling_number
    Tb = unitify(resampling_period)
    Nb = resampling_number
    Nea, Neb = lmn.convolution_downsample_size(Ta, Na, Tb, Nb)
    print(f'(Ta={Ta/units.ns:.1f} ns, {Na=}) -> {Nea}, (Tb={Tb/units.ns:.1f} ns, {Nb=}) -> {Neb}')


@cli.command("lmn")
@click.option("-n", "--sampling-number", default=None, type=int,
              help="Original number of samples.")
@click.option("-t", "--sampling-period", default=None, type=str,
              help="Original sample period,eg '100*ns'")
@click.option("-T", "--resampling-period", default=None, type=str,
              help="Resampled sample period, eg '64*ns'")
@click.option("-e", "--error", default=1e-6,
              help="Precision by which integer and "
              "rationality conditions are judged")
def cmd_lmn(sampling_number, sampling_period, resampling_period, error):
    '''Print various LMN parameters for a given resampling.

    '''
    Ns, Ts, Tr = sampling_number, sampling_period, resampling_period

    if not Ts or not Ns or not Tr:
        raise click.BadParameter('Must provide all of -n, -t and -T')

    Ts = unitify(Ts)
    Tr = unitify(Tr)

    from wirecell.util import lmn
    print(f'initial sampling: {Ns=} {Ts=}')

    gcd = lmn.egcd(Tr, Ts-Tr)
    print(f'egcd({Ts=}, {Tr=}): {gcd}')
    dn = lmn.rational_deltan(Ts, Tr)
    print(f'minimum delta-n: {dn}')
    nrat = lmn.rational_size(Ts, Tr, error)
    print(f'minimum size: {nrat}')

    nrag = Ns % nrat
    if nrag:
        npad = nrat - nrag
        Ns_rational = Ns + npad
        print(f'rationality padding: {Ns=} += {npad=} -> {Ns_rational=}')
    else:
        print(f'rationality met: {Ns=}')
        Ns_rational = Ns

    Nr_target = Ns_rational*Ts/Tr
    assert abs(Nr_target - round(Nr_target)) < error
    Nr_target = round(Nr_target)
    print(f'resampling target: {Nr_target=} {Tr=}')

    ndiff = Nr_target - Ns_rational
    print(f'final resampling: {Ns_rational=}, {Ts=} '
          f'-> {Nr_target=}, {Tr=} diff of {ndiff}')

    Nr_wanted = Ns*Ts/Tr
    Nr = math.ceil(Nr_wanted)
    if abs(Nr_wanted - Nr) > error:
        print('--> note: noninteger nominal target size:'
              f' {Nr_wanted} for {Ns=} {Ts=} {Tr=}.')
    print(f'resampling target size: {Nr=} {Tr=}')

    ntrunc = Nr - Nr_target
    if ntrunc < 0:
        print(f'truncate resampled: {Nr_target} -> {Nr}, remove {-ntrunc}')
    if ntrunc > 0:
        print(f'extend resampled: {Nr_target} -> {Nr}, insert {ntrunc}')



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
@click.option("--face-style", default=None,
              help="Set convention for face per anode (eg 'sbnd').")
@click.argument("input-file")
@click.argument("output-file")
@click.pass_context
def convert_multitpc_wires(ctx, face_style, input_file, output_file):
    '''
    Convert a "multitpc" wire description file into one suitable for
    WCT.

    Here "TPC" refers to on anode face.  That is, one APA is made up
    of two TPCs.  An example file is protodune-wires-larsoft-v1.txt
    from wire-cell-data.  It has columns like:

        # chan tpc plane wire sx sy sz ex ey ez

    And, further the order of rows of identical channel number express
    progressively higher segment count.

    (Optional) "--face-style" allows user to choose a different naming
    convention for numbering face. Currently, only "sbnd" is supported.
    '''
    from wirecell.util.wires import multitpc, persist
    store = multitpc.load(input_file, face_style=face_style)
    persist.dump(output_file, store)

@cli.command("convert-icarustpc-wires")
@click.argument("input-file")
@click.argument("output-file")
@click.pass_context
def convert_icarustpc_wires(ctx, input_file, output_file):
    '''
    Description
    '''
    from wirecell.util.wires import icarustpc, persist
    store = icarustpc.load(input_file)
    persist.dump(output_file, store)

@cli.command("convert-dunevd-wires")
@click.option('-t', '--type', default="3view",
              help='2view, 3view, 3view_30deg, coldbox, protodunevd, protodunevd_drifty')
@click.argument("input-file")
@click.argument("output-file")
@click.pass_context
def convert_dunevd_wires(ctx, type, input_file, output_file):
    '''
    Convert txt wire geom to json format
    '''
    from wirecell.util.wires import dunevd, persist
    if type == 'protodunevd' or type == 'protodunevd_drifty':
        input_file = dunevd.merge_tpc(input_file)
    store = dunevd.load(input_file, type)
    persist.dump(output_file, store)

@cli.command("convert-uboone-wire-regions")
@click.argument("wire-json-file")
@click.argument("csvfile")
@click.argument("region-json-file")
@click.pass_context
def convert_uboon_wire_regions(ctx, wire_json_file, csvfile, region_json_file):
    '''
    Convert CSV file to WCT format for wire regions.  Example is one
    as saved from MicroBooNE_ShortedWireList.xlsx.  Use ,-separated
    and remove quoting.
    '''
    import wirecell.util.wires.persist as wpersist
    import wirecell.util.wires.regions as reg
    store = wpersist.load(wire_json_file)
    ubs = reg.uboone_shorted(store, csvfile)
    wpersist.dump(region_json_file, ubs)

@cli.command("plot-wire-regions")
@click.argument("wire-json-file")
@click.argument("region-json-file")
@click.argument("pdf-file")
@click.pass_context
def plot_wire_regions(ctx, wire_json_file, region_json_file, pdf_file):
    import wirecell.util.wires.persist as wpersist
    import wirecell.util.wires.plot as wplot
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    store = wpersist.load(wire_json_file)
    regions = wpersist.load(region_json_file)


    def pt2xy(pt):
        'Point id to xy tuple'
        ptobj = store.points[pt]
        return (ptobj.z, ptobj.y)
    def wo2pg(wo1,wo2):
        'wire objects to polygon'
        return numpy.asarray([pt2xy(wo1.tail),pt2xy(wo1.head),
                              pt2xy(wo2.head),pt2xy(wo2.tail)])

    colors=['red','green','blue']

    def get_polygons(shorted, triples):
        ret = list()
        for trip in triples:

            for one in trip:
                pl,wip1,wip2 = one["plane"],one["wire1"],one["wire2"]
                if pl != shorted:
                    continue

                # fixme: this line assumes only 1 face
                plobj = store.planes[pl]
                wobj1 = store.wires[plobj.wires[wip1]]
                wobj2 = store.wires[plobj.wires[wip2]]

                assert wobj1.channel == one['ch1']
                assert wobj2.channel == one['ch2']

                verts = wo2pg(wobj1,wobj2)
                pg = Polygon(verts, closed=True, facecolor=colors[pl],
                             alpha=0.3, fill=True, linewidth=.1, edgecolor='black')
                ret.append(pg)
        return ret

    pgs = [get_polygons(int(s), t) for s,t in regions.items()]
    pgs2 = [get_polygons(int(s), t) for s,t in regions.items()]
    pgs.append(pgs2[0] + pgs2[1])

    zlimits = [ (0,4100), (6900,7500), (0, 7500) ]

    with PdfPages(pdf_file) as pdf:

        for pgl,zlim in zip(pgs,zlimits):
            fig, ax = plt.subplots(nrows=1, ncols=1)
            for pg in pgl:
                ax.add_patch(pg)

            ax.set_xlim(*zlim)
            ax.set_ylim(-1500,1500)
            ax.set_title('Dead wires')
            ax.set_xlabel('Z [mm]')
            ax.set_ylabel('Y [mm]')
            pdf.savefig(fig)
            plt.close()
    return


@cli.command("wires-info")
@click.argument("json-file")
@click.pass_context
def wires_info(ctx, json_file):
    '''
    Print information about a wires file (.json or .json.bz2)
    '''
    import wirecell.util.wires.persist as wpersist
    import wirecell.util.wires.info as winfo
    wires = wpersist.load(json_file)
    dat = winfo.summary(wires)
    log.info ('\n'.join(dat))


@cli.command("wires-ordering")
@click.option("-o", "--output", default=None,
              help="Output file, default based on input")
@click.argument("json-file")
@click.pass_context
def wires_ordering(ctx, output, json_file):
    '''
    Make PDF showing info on how wires are ordered in each plane.

    The result is for debugging and perhaps cryptic.
    '''
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(nrows=2,ncols=1)

    bname = os.path.basename(json_file)
    name = bname[:bname.find(".json")]
    if output is None:
        output = name + "-wire-ordering.pdf"
    plt.suptitle(name)
    titles = ["Y", "Z", "S", "Zi"] # match calls to add_plot()

    def add_plot(ind, arr, lab):
        da = arr[1:] - arr[:-1]
        pa = numpy.zeros_like(da);
        pa[da>0] = +1.0;
        pa[da<0] = -1.0;
        axes[ind].plot(pa, label=lab)

    import wirecell.util.wires.persist as wpersist
    import wirecell.util.wires.info as winfo
    wires = winfo.todict(wpersist.load(json_file))
    for anode in wires[0]["anodes"][:1]:
        aid=anode["ident"]
        for face in anode["faces"]:
            fid=face["ident"]
            for plane in face["planes"]:
                pid=plane["ident"]
                dat=defaultdict(list)

                wires = plane["wires"]
                h = numpy.array([tuple(w["head"].values()) for w in wires])
                t = numpy.array([tuple(w["tail"].values()) for w in wires])
                # shape is (nwires,3)
                m = 0.5*(h+t)   # midpoints
                x,y,z = m.T
                ymin = numpy.min(y)
                zmin = numpy.min(z)
                yr = y - ymin
                zr = z - zmin
                dw = h - t      # wire length vectors
                mw = numpy.sqrt(numpy.sum(numpy.abs(dw)**2,axis=-1))
                w = dw/mw[:,None]        # wire unit vectors
                Y = numpy.array((0,1,0)) # y axis
                Z = numpy.array((0,0,1)) # z axis
                s = yr*numpy.abs(numpy.dot(w,Z)) + zr*numpy.abs(numpy.dot(w,Y))

                add_plot(0,y,f'Y a={aid} f={fid} p={pid}')
                add_plot(1,z,f'Z a={aid} f={fid} p={pid}')
                        
    log.info(output)
    for ind, ax in enumerate(axes):
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax.set_title(titles[ind])
    plt.tight_layout()
    fig.savefig(output)
        

@cli.command("wires-channels")
@click.option("-o", "--output", default=None,
              help="Output file, default based on input")
@click.argument("json-file")
@click.pass_context
def wires_channels(ctx, output, json_file):
    '''
    Make PDF showing info on how wires vs channels.
    '''
    import wirecell.util.wires.persist as wpersist
    import wirecell.util.wires.info as winfo
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    titles = ["X vs ch", "Y vs ch", "Z vs ch"]

    bname = os.path.basename(json_file)
    name = bname[:bname.find(".json")]
    if output is None:
        output = name + "-wires-channels.pdf"


    all_wires = winfo.todict(wpersist.load(json_file))

    specs = ["center", "head", "tail"]
    def get_xyz(wires, spec):
        if spec in ("head","center"):
            h = numpy.array([tuple(w["head"].values()) for w in wires])
        if spec in ("tail","center"):
            t = numpy.array([tuple(w["tail"].values()) for w in wires])
        if spec == "center":
            m = 0.5*(h+t)   # midpoints
            return m.T
        if spec == "head":
            return h.T
        if spec == "tail":
            return t.T
        raise ValueError(f'spec {spec} not supported')
    
    with PdfPages(output) as pdf:

        for spec in specs:

            fig,axes = plt.subplots(nrows=len(titles),ncols=1)
            plt.suptitle(name)

            for anode in all_wires[0]["anodes"][:1]:
                aid=anode["ident"]
                for face in anode["faces"]:
                    fid=face["ident"]
                    for plane in face["planes"]:
                        pid=plane["ident"]
                        dat=defaultdict(list)

                        pwires = plane["wires"]
                        x,y,z = get_xyz(pwires, spec)
                        c = numpy.array([w["channel"] for w in pwires])

                        ptype="scatter"
                        par = dict(marker="o")
                        if ptype == "scatter":
                            par["s"] = 0.2
                        getattr(axes[0], ptype)(c, x, label=f'X a={aid} f={fid} p={pid}',**par)
                        getattr(axes[1], ptype)(c, y, label=f'Y a={aid} f={fid} p={pid}',**par)
                        getattr(axes[2], ptype)(c, z, label=f'Z a={aid} f={fid} p={pid}',**par)

            for ind, ax in enumerate(axes):
                ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                tit = titles[ind]
                ax.set_title(f'{spec}: {tit}')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
    log.info(output)

    

@cli.command("wires-volumes")
@click.option('-a', '--anode', default="1*cm",
              help='Distance from collection plane to "anode" (cutoff) plane (in system of units)')
@click.option('-r', '--response', default="10*cm",
              help='Distance from collection plane to "respones" plane, should probably match Garfield (in system of units)')
@click.option('-c', '--cathode', default="1*m",
              help='Distance from colleciton plane to "cathode" plane (in system of units)')
@click.argument("json-file")
@click.pass_context
def wires_volumes(ctx, anode, response, cathode, json_file):
    '''
    Print a params.geometry JSON fragment for the given wires file.

    You likely want to pipe this through jsonnetfmt go get pretty print.
    '''
    import wirecell.util.wires.info as winfo
    jv = winfo.jsonnet_volumes(json_file, unitify(anode), unitify(response), unitify(cathode))
    click.echo(str(jv))



@cli.command("plot-wires")
@click.option('-w', '--wire-step', default=10,
               help='Number of wires to skip in the wire plots (default: 10)')
@click.option('-d', '--drift-axis', default=0,
               help='Index of the drift direction [0 (default): x-axis, 1: y-axis]')
@click.argument("json-file")
@click.argument("pdf-file")
@click.pass_context
def plot_wires(ctx, json_file, pdf_file, wire_step, drift_axis):
    '''
    Plot wires from a WCT JSON(.bz2) wire file
    '''
    import wirecell.util.wires.persist as wpersist
    import wirecell.util.wires.plot as wplot
    wires = wpersist.load(json_file)
    wplot.allplanes(wires, pdf_file, wire_step, drift_axis)


@cli.command("plot-select-channels")
@click.option('--labels/--no-labels', default=True,
              help="Use labels or not")
@click.argument("json-file")
@click.argument("pdf-file")
@click.argument("channels", nargs=-1, type=int)
@click.pass_context
def plot_select_channels(ctx, labels, json_file, pdf_file, channels):
    '''
    Plot wires for select channels from a WCT JSON(.bz2) wire file
    '''
    import wirecell.util.wires.persist as wpersist
    import wirecell.util.wires.plot as wplot
    wires = wpersist.load(json_file)
    wplot.select_channels(wires, pdf_file, channels, labels=labels)


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
    fig.savefig(output_file)

@cli.command("make-wires")
@click.option('-d','--detector',
#              type=click.Choice("microboone protodune dune apa".split()),
              default="apa",
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
        store = graph.to_schema(G, P, apa.channel_ident)
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
@click.option("--width", default=None, type=str,
              help="Detector width")
@click.option("--height", default=None, type=str,
              help="Detector height")
@click.option("--pitches", default=None, type=str,
              help="Comma list of pitch distances")
@click.option("--angles", default=None, type=str,
              help="Comma list of angles")
@click.option("--offsets", default=None, type=str,
              help="Comma list of wire offsets")
@click.option("--planex", default=None, type=str,
              help="Comma list of plane X locations")
@click.option("--mcpp", default=None, type=int,
              help="Maximum number of channels in a plane")
@click.option("-d", "--detector",
              default="protodune",
              type=click.Choice(["protodune", "microboone"]),
              help="Set default parameter set")
@click.argument("output-file")
@click.pass_context
def make_wires_onesided(ctx, **kwds):
    '''
    Generate a WCT wires file.

    The named detector sets initial defaults.  If other options are
    given they overwride.
    '''
    output_file = kwds.pop("output_file")
    detector = kwds.pop("detector")

    import wirecell.util.wires.generator as wgen
    params = dict(getattr(wgen, f'{detector}_params'))

    for key, val in kwds.items():
        if val is None:
            continue
        if isinstance(val, str):
            val = unitify_parse(val)
            if len(val) == 1:
                val = val[0]
        params[key] = val

    import wirecell.util.wires.persist as wpersist
    s = wgen.onesided_wrapped(params)
    wpersist.dump(output_file, s)

@cli.command("wire-channel-map")
@click.argument("input-file")
@click.pass_context
def wire_channel_map(ctx, input_file):
    '''
    Debug command, generate a WCT channel map wires file.
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
            click.echo("%s\t%s" %(c,wires))

@cli.command("wire-summary")
@jsonnet_loader("wires", "wires")
@click.option("-o", "--output", default="/dev/stdout",
              help="Output file")
def wire_summary(output, wires):
    '''
    Produce a summary of detector response and wires.
    '''
    import wirecell.util.wires.info as winfo
    import wirecell.util.wires.persist as wper

    # wash initial dict through schema to assure it is valid
    store = wper.fromdict(wires)
    wdict = winfo.summary_dict(store)
    with open(output, "w") as fp:
        fp.write(json.dumps(wdict, indent=4))



@cli.command('frame-split')
@click.option("-r", "--rebin", default=0, type=int,
              help="Rebin the time/colums/X-axis by this many bins")
@click.option("-t", "--tick-offset", default=0, type=int,
              help="Shift the content by this many bins on time/colums/X-axis (prior to any rebin)")
@click.option("-f", "--fpattern",
              default=None,
              help="Set the format for the output files, def: adds .npz to apattern")
@click.option("-a", "--apattern",
              default="{detector}-{tag}-{index}-{anodeid}-{planeid}",
              help="Set the format for the array name")
@click.option("-m", "--metadata",
              default=None,
              help="Additional metadata as JSON file to apply to patterns and save")
@click.argument("archive")
def frame_split(rebin, tick_offset, fpattern, apattern, metadata, archive):
    '''
    Split an archive of frame arays into per-plane Numpy .npz files.

    The archive file may be one of:

        - .npz :: file as saved by NumpyFrameSaver

        - .tar :: file as saved by FrameFileSink (or .tar.gz,
          .tar.bz2)

    Available variables for the format are:

        - detector :: the detector name devined from the input.

        - tag :: the frame tag from the input, empty implies "orig".

        - index :: the numeric cound from the array name

        - anodeid :: the numberic ID of the anode

        - planeid :: the numeric ID of the plane (0=U, 1=V, 2=W)

        - planeletter :: letter for the plane

        - rebin :: amount of rebin or zero
    '''
    
    from .frame_split import guess_splitter, save_one
    from . import ario

    if metadata:
        metadata = json.loads(open(metadata).read())
    else:
        metadata = dict()

    fp = ario.load(archive)
    for aname in fp.keys():
        if not aname.startswith("frame_"):
            continue
        frame = fp[aname]
        meth = guess_splitter(frame.shape[0])

        parts = aname.split("_")
        log.debug(f'splitting: {parts}')
        tag = parts[1] or "orig"
        index = int(parts[2])

        gots = meth(frame, tag, index, tick_offset, rebin)
        for arr, md in gots:

            md.update(**metadata)

            aname = apattern.format(**md)
            if fpattern:
                fpath = fpattern.format(**md)
            else:
                fpath = aname + ".npz"

            save_one(fpath, aname, arr, md)
        

@cli.command("npz-to-img")
@click.option("-o", "--output", type=str,
              help="Output image file")
@click.option("-a", "--array", type=str, default=None,
              help="Array to plot")
@click.option("-c", "--cmap", type=str, default="seismic",
              help="Color map")
@click.option("-b", "--baseline", default=None,
              help="Apply median baseline subtraction or if number, subtract directly")
@click.option("-m", "--mask", default=None, 
              help="Value or range to mask, range is lower-edge inclusive")
@click.option("--vmin", default=None, 
              help="Minimum value (z/color axis)")
@click.option("--vmax", default=None, 
              help="Maximum value (z/color axis)")
@click.option("--dpi", default=None,
              help="The dots-per-inch resolution")
@click.option("-z", "--zoom", default=None,
              help="A zoom range as 'rmin:rmax,cmin:cmax'")
@click.option("-X", "--xtitle", default=None,
              help="X-axis title")
@click.option("-Y", "--ytitle", default=None,
              help="Y-axis title")
@click.option("-Z", "--ztitle", default=None,
              help="Z-axis title")
@click.option("-T", "--title", default=None,
              help="Overall title")
@click.argument("npzfile")
def npz_to_img(output, array,
               cmap, baseline, mask, vmin, vmax, dpi, zoom,
               xtitle, ytitle, ztitle, title,
               npzfile):
    '''
    Make an image from an array in an numpy file.
    '''
    # fixme: expose options, aspect, pcolor, colobar, to cli

    import matplotlib.pyplot as plt

    if not output:
        raise click.BadParameter("need output file")

    fp = numpy.load(npzfile)
    if not array:
        array = list(fp.keys())[0]
    arr = fp[array]

    if baseline == "median":
        arr = (arr.T - numpy.median(arr, axis=1)).T
    elif baseline is not None:
        arr = arr - float(baseline)

    args = dict(cmap=cmap, aspect='auto', interpolation='none')

    if vmin is not None:
        args["vmin"] = vmin
    if vmax is not None:
        args["vmax"] = vmax
    if mask:
        mask = [float(m) for m in mask.split(",")]
        arr = numpy.ma.array(arr)
        if len(mask) == 1:
            arr = numpy.ma.masked_where(arr == mask[0], arr)
        else:
            arr = numpy.ma.masked_where((arr >= mask[0]) & (arr < mask[1]), arr)
    if zoom:
        r,c = zoom.split(",")
        rslc = slice(*[int(i) for i in r.split(":")])
        cslc = slice(*[int(i) for i in c.split(":")])
        arr = arr[rslc, cslc]

    if not arr.shape[0] or not arr.shape[1]:
        raise ValueError("Array became zero")

    plt.imshow(arr, **args)
    cb = plt.colorbar(label=ztitle)

    if xtitle:
        plt.xlabel(xtitle)
    if ytitle:
        plt.ylabel(ytitle)
    if title:
        plt.suptitle(title)

    args = dict()
    if dpi:
        args["dpi"] = int(dpi)
    plt.savefig(output, **args)

@cli.command("ls")
@click.argument("filename")
def ls(filename):
    '''
    List contents of a WCT file (.npz, .zip or .tar[.gz])
    '''
    from . import ario, tdm, cdm
    fp = ario.load(filename)
    if tdm.looks_like(fp):
        log.info(tdm.dumps(fp))
        return
    if cdm.looks_like(fp):
        log.info(cdm.dumps(fp))
        return
    for key, val in fp.items():
        if isinstance(val, numpy.ndarray):
            log.info(f'{key} {val.shape} {val.dtype}')
        else:
            log.info(f'{key} {type(val)}')
    
@cli.command('pc2pd')
@click.option("--ident", default=None, type=int,
              help="Limit loading to a specific tensorset of given ident (default converts all)")
@click.option("--prefix", default="",
              help='Limit loading to tdm entries starting with a prefix prefix (default="")')
@click.option("--points", default=None, required=True,
              help='Comma-separated list of datapath to arrays to use as Cartesian points')
@click.option("--attrs", default=None, 
              help='Comma-sparated list of datapath to arrays to limit as atributes, default is all')
@click.argument('pcfile')
@click.argument('pdfile')
def pc2pd(ident, prefix, points, attrs, pcfile, pdfile):
    '''
    Convert pointcloud in WCT TDM file format to VTK PolyData file.

    '''
    if not pdfile.endswith(".vtp"):
        log.warning(f'output file name does not end in .vtp, paraview may complain')        

    from . import ario, tdm
    try:
        from tvtk.api import tvtk, write_data
    except ImportError:
        raise click.ClickException('''no suport for tvtk
        try:  apt install mayavi2
        or:   pip install mayavi
        depending on your context''')

    tens = tdm.load(ario.load(pcfile), prefix, ident)
    if tens is None:
        raise click.ClickException(f'failed to load tensors at {prefix=}, {ident=} from {pcfile}')
    if isinstance(tens, tdm.Tree):
        tens = [tens]

    points = points.split(',')
    attrs = attrs.split(',') if attrs else []

    def get_tensorset(top):
        pcds = top.visit_by_metadata(datatype='pcdataset')
        if not pcds:
            return
        ret = dict()
        for key, path in pcds[0].md['arrays'].items():
            ret[key] = top(path)
        return ret

    ptens = defaultdict(list)
    atens = defaultdict(list)

    # collect arrays
    for count, ten in enumerate(tens):
        ts = get_tensorset(ten)
        if not ts:
            log.warning(f'tree {count} has no tensor set')
            continue

        if not attrs:
            attrs = [n for n in ts if n not in points]

        okay = True
        for name in points + attrs:
            if name not in ts:
                log.warning(f'tensor set {count} lacks tensor {name}, skipping.\nUsing --prefix/--ident may avoid this message.')
                okay = False
            if ts[name].array is None:
                log.warning(f'tensor set {count} tensor {name} lacks array, skipping.\nUsing --prefix/--ident may avoid this message.')
                okay = False
        if not okay:
            continue
            
        for name in points:
            ptens[name].append(ts[name].array)
        for name in attrs:
            atens[name].append(ts[name].array)

    log.info(f'{points=} {attrs=}')

    point_arrays = [numpy.hstack(ptens[name]) for name in points]
    attrs_arrays = dict()
    for name in atens:
        attrs_arrays[name] = numpy.hstack(atens[name])

    pd = tvtk.PolyData()
    tdm.pc2vtk(pd, *point_arrays, **attrs_arrays)
    write_data(pd, pdfile)

@cli.command('tdm2hdf')
@click.option("--ident", default=None, type=int,
              help="Convert a specific tensorset of given ident (default converts all)")
@click.option("--prefix", default="",
              help='The tdm entry prefix (default="")')
@click.argument('tdmfile')
@click.argument('hdffile')
def tdm2hdf(ident, prefix, tdmfile, hdffile):
    '''
    Convert WCT TDM file to HDF equivalent.
    '''
    from . import ario, tdm
    try:
        import h5py
    except ImportError:
        raise click.ClickException('''no suport for h5py
        try:  apt install python3-h5py
        or:   pip install h5py
        depending on your context''')

    tens = tdm.load(ario.load(tdmfile), prefix, ident)
    if tens is None:
        raise click.ClickException(f'failed to load tensors at {prefix=}, {ident=} from {tdmfile}')
    log.debug(f'loaded {len(tens)} tensors')
    tdm.tohdf(h5py.File(hdffile, 'w'), tens)


@cli.command("dump-tdm")
@click.option("-i", "--index", default=0, help="Index of tensorset (def=0)")
@click.argument("filename")
def dump_tdm(index, filename):
    '''
    Dump file in WCT tensor-data-model form.
    '''
    from . import ario, tdm
    fp = ario.load(filename)
    if not tdm.looks_like(fp):
        click.echo("file does not appear to hold a tensor set")
        return 1

    def v(n,c):
        pre = '\t'*len(c)
        loc = '/'.join(c)
        keys = str(list(n.md.keys()))
        return f'{loc}\n{pre}md:{keys}'

    for one in tdm.load(fp):
        got = one.visit(v, with_context=True)
        print('\n'.join(got))
        


@cli.command("npz-to-wct")
@click.option("-T", "--transpose", default=False, is_flag=True,
              help="Transpose input arrays to give per-channel rows")
@click.option("-o", "--output", type=str,
              help="Output image file")
@click.option("-n", "--name", default="",
              help="The name tag for the output arrays")
@click.option("-f", "--format", default="frame",
              type=click.Choice(["frame",]), # "tensor"
              help="Set the output file format")
@click.option("-t", "--tinfo", type=str,
              default="0,0.5*us,0",
              help="The tick info list: time,tick,tbin")
@click.option("-b", "--baseline", default=0.0,
              help="An additive, prescaled offset")
@click.option("-s", "--scale", default=1.0,
              help="A multiplicative scaling")
@click.option("-d", "--dtype", default="i2",
              type=click.Choice(["i2","f4"]),
              help="The data type of output samples in Numpy dtype form")
@click.option("-c", "--channels", default=None,
              help="Channel specification")
@click.option("-e", "--event", default=0,
              help="Event count start")
@click.option("-z", "--compress", default=True, is_flag=True,
              help="Whether to compress if output file is .npz")
@click.argument("npzfile")
def npz_to_wct(transpose, output, name, format, tinfo, baseline, scale, dtype, channels, event, compress, npzfile):
    """Convert a npz file holding 2D frame array(s) to a file for input to WCT.

    A linear transform and type cast is be applied to the input
    samples prior to output:

        output = dtype((input + baseline) * scale)

    Channel ID numbers for rows of the input array must be specified
    in a way to that matches the target detector.  They may be
    specified in a number of ways:

    - Default (unspecified) will number them starting at ID=0.
    - A single integer N will number them starting at ID=N.
    - A comma-separated list: 1,2,3,.... exaustively gives all IDs.
    - A file.npy with a 1D array of integers.
    - A file.npz:array_name with a 1D array of integers.

    """
    from collections import OrderedDict

    tinfo = unitify(tinfo)
    baseline = float(baseline)
    scale = float(scale)

    out_arrays = OrderedDict()
    event = int(event)          # count "event" number
    fp = numpy.load(npzfile)
    for aname in fp:
        arr = fp[aname]
        if transpose:
            arr = arr.T
        if len(arr.shape) != 2:
            raise click.BadParameter(f'input array {aname} wrong shape: {arr.shape}')

        nchans = arr.shape[0]

        # figure out channels in the loop as nchans may differ array
        # to array.
        if channels is None:
            channels = list(range(nchans))
        elif channels.isdigit():
            ch0 = int(channels)
            channels = list(range(ch0, ch0+nchans))
        elif "," in channels:
            channels = unitify(channels)
            if len(channels) != nchans:
                raise click.BadParameter(f'input array has {nchans} channels but given {len(channels)} channels')

        elif channels.endswith(".npy"):
            channels = numpy.load(channels)
        elif ".npz:" in channels:
            fname,cname = channels.split(":",1)
            cfp = numpy.load(fname)
            channels = cfp[cname]
        else:
            raise click.BadParameter(f'unsupported form for channels: {channels}')

        channels = numpy.array(channels, 'i4')

        label = f'{name}_{event}'
        event += 1

        out_arrays[f'frame_{label}'] = numpy.array((arr + baseline) * scale, dtype=dtype)
        out_arrays[f'channels_{label}'] = channels
        out_arrays[f'tickinfo_{label}'] = tinfo

    if output.endswith(".npz"):
        if compress:
            numpy.savez_compressed(output, **out_arrays)
        else:
            numpy.savez(output, **out_arrays)
    else:
        raise click.BadParameter(f'unsupported output file type: {output}')


@cli.command("ario-cmp")
@click.option("-f","--filenames", type=str, default=None,
              help="comma-separated list of file names in archives to use, default is all")
@click.argument("ario1")
@click.argument("ario2")
def ario_cmp(filenames, ario1, ario2):
    '''Return true exit code if two ario archives compare the same.

    An ario archive is any file supported by wirecell.util.ario.

    Archives differ if any of their element file names differ or if
    matched file names have different content.
    '''
    from . import ario
    a1 = ario.load(ario1)
    a2 = ario.load(ario2)

    keys1 = list(a1.keys())
    keys2 = list(a2.keys())

    if filenames:
        keys = [ario.stem_if(n, ('npy', 'json')) for n in filenames.split(",")]
    else:
        keys = list(set(keys1 + keys2))
    keys.sort()
    log.debug(f'{keys1}, {keys2}')

    def are_same(key):
        d1 = a1[key];  d2 = a2[key]
        t1 = type(d1); t2 = type(d2) 
        if t1 != t2:
            log.debug(f'{key} type mismatch: {t1} and {t2}')
            return False
        if isinstance(d1, numpy.ndarray):
            if d1.dtype != d2.dtype:
                log.debug(f'{key} array dtype mismatch: {d1.dtype} and {d2.dtype}')
                return False
            if d1.shape != d2.shape:
                log.debug(f'{key} array shape mismatch: {d1.shape} and {d2.shape}')
                return False
            if not numpy.array_equal(d1, d2):
                log.debug(f'{key} array value mismatch')
                return False
            return True
        if d1 != d2:
            log.debug(f'{key} object value mismatch')
            return False
        return True

    ndiffs = 0
    for key in keys:
        have1 = key in keys1
        have2 = key in keys2
        if have1 and have2:
            if are_same(key):
                ndiffs += 1
            continue
        if have1:
            log.debug(f'{key} only in {ario1}')
            ndiffs += 1
            continue
        if have2:
            log.debug(f'{key} only in {ario2}')
            ndiffs += 1
            continue
        log.debug(f'{key} not found')
        continue
    sys.exit(ndiffs)


@cli.command("detectors")
@click.option("-p","--path", default=(), multiple=True,
              help="Add a search path")
def cmd_detectors(path):
    '''
    Known canonical detectors. 
    '''
    import sys
    path = list(path) + list(wirecell_path())
    sys.stderr.write(f'searching: {path}\n')
    dets = jsio.resolve("detectors.jsonnet", path)
    sys.stderr.write(f'detectors file: {dets}\n')
    dets = jsio.load(dets, path)
    print (' '.join(dets.keys()))
        


@cli.command("resample")
@click.option("-t", "--tick", default='500*ns',
              help="Resample the frame to have this sample period with units, eg '500*ns'")
@click.option("-o","--output", type=str, required=True,
              help="Output filename")
@click.argument("framefile")
def resample(tick, output, framefile):
    '''
    Resample a frame file
    '''
    from . import ario, lmn

    Tr = unitify(tick)
    print(f'resample to {Tr/units.ns}ns to {output}')


    fp = ario.load(framefile)
    f_names = [k for k in fp.keys() if k.startswith("frame_")]
    c_names = [k for k in fp.keys() if k.startswith("channels_")]
    t_names = [k for k in fp.keys() if k.startswith("tickinfo_")]

    out = dict()

    for fnum, frame_name in enumerate(f_names):
        _, suffix = frame_name.split('_',1)
        ti = fp[f'tickinfo_{suffix}']
        Ts = ti[1]

        if Tr == Ts:
            print(f'frame {fnum} "{frame_name}" already sampled at {Tr}')
            continue

        frame = fp[frame_name]
        Ns = frame.shape[1]
        ss = lmn.Sampling(T=Ts, N=Ns)

        Nr = round(Ns * Tr / Ts)
        sr = lmn.Sampling(T=Tr, N=Nr)

        print(f'{fnum} {frame_name} {ss=} -> {sr=}')

        resampled = numpy.zeros((frame.shape[0], Nr), dtype=frame.dtype)
        for irow, row in enumerate(frame):
            sig = lmn.Signal(ss, wave=row)
            resig = lmn.interpolate(sig, Tr)
            wave = resig.wave
            # if Nr != wave.size:
            #     print(f'resizing to min({Nr=},{wave.size=})')
            Nend = min(Nr, wave.size)
            resampled[irow,:Nend] = wave[:Nend]
        
        out[f'frame_{suffix}'] = resampled
        out[f'tickinfo_{suffix}'] = numpy.array([ti[0], Tr, ti[2]])
        out[f'channels_{suffix}'] = fp[f'channels_{suffix}']

    numpy.savez_compressed(output, **out)

@cli.command("resolve")
@click.option("-p","--path", default=(), multiple=True,
              help="Add a search path")
@click.option("-k","--kind", default=None,
              help="If name gives a detector then kind must give what kind of file")
@click.argument("name")
def resolve(path, kind, name):
    '''Resolve name to a WCT file '''
    path = list(path) + list(wirecell_path())
    
    def emit(got):
        if not isinstance(got, list):
            got = [got]
        for one in got:
            if isinstance(one, str):
                print(one)
            else:
                print(one.absolute())

    # fixme: these functions should be in a more generic module!
    path = jsio.clean_paths(path)
    try:
        got = jsio.resolve(name, path)
    except RuntimeError:
        pass
    else:
        emit(got)
        return

    if kind:
        try:
            got = detectors.resolve(name, kind)
        except KeyError:
            pass
        else:
            emit(got)
            return

    sys.stderr.write(f'failed to find {name}, considered paths:\n')
    sys.stderr.write('\n\t'.join(path))
    sys.stderr.write('\n')
    if kind:
        sys.stderr.write(f'and {kind=}\n')
    raise ValueError('bad fields')


@cli.command("framels")
@click.option("-o", "--output", default="/dev/stdout", help="Output filename")
@click.argument("framefile")
def framels(output, framefile):
    '''
    Print information about a frame file
    '''
    f = numpy.load(framefile)

    # fixme: make more flexible as for order.

    assert f.files[0].startswith("frame_")
    fr = f[f.files[0]]
    _,tag,ident = f.files[0].split("_")

    assert f.files[1].startswith("channels_")
    ch = f[f.files[1]]

    assert f.files[2].startswith("tickinfo_")
    ti = f[f.files[2]]

    summary = dict(tag=tag, ident=int(ident),
                   shape=fr.shape,
                   chmin=int(numpy.min(ch)), chmax=int(numpy.max(ch)),
                   t0=ti[0], tick=ti[1], tbin=int(ti[2]))
    open(output,"w").write(json.dumps(summary, indent=4) + "\n")

def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
