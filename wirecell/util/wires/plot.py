
from wirecell import units
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy
from collections import defaultdict
import math

def plot_polyline(pts):
    cmap = plt.get_cmap('seismic')
    npts = len(pts)
    colors = [cmap(i) for i in numpy.linspace(0, 1, npts)]
    for ind, (p1, p2) in enumerate(zip(pts[:-1], pts[1:])):
        x = numpy.asarray((p1.x, p2.x))
        y = numpy.asarray((p1.y, p2.y))
        plt.plot(x, y,  linewidth=ind+1)
    


def oneplane(store, iplane, segments=None):
    '''
    Plot one plane of wires.

    This plot is in protodune-numbers document.
    '''
    fig,axes = plt.subplots(nrows=1, ncols=3)

    uvw = "UVW"

    widths = [1, 2, 3]
    wire_stride=20;

    iface = 0
    face = store.faces[iface]

    cmap = plt.get_cmap('rainbow')

    for iplane in range(3):
        ax = axes[iplane]

        planeid = face.planes[iplane]
        plane = store.planes[planeid]

        wires = [w for w in plane.wires[::wire_stride]]

        nwires = len(wires)
        colors = [cmap(i) for i in numpy.linspace(0, 1, nwires)]

        for wcount, wind in enumerate(wires):

            wire = store.wires[wind]
            if segments and not wire.segment in segments:
                continue

            color = colors[wcount]
            if not iplane:
                color = colors[nwires-wcount-1]

            p1 = store.points[wire.tail]
            p2 = store.points[wire.head]
            width = widths[wire.segment]
            ax.plot((p1.z/units.meter, p2.z/units.meter), (p1.y/units.meter, p2.y/units.meter),
                        linewidth = width, color=color)

            ax.locator_params(axis='x', nbins=5)
            ax.set_aspect('equal', 'box')
            ax.set_xlabel("Z [meter]")
            if not iplane:
                ax.set_ylabel("Y [meter]")
            ax.set_title("plane %d/%s" % (iplane,uvw[iplane]))

    return fig,ax

def select_channels(store, pdffile, channels, labels=True):
    '''
    Plot wires for select channels.
    '''
    channels = set(channels)
    bychan = defaultdict(list)

    # find selected wires and their wire-in-plane index
    # fixme: there should be a better way!
    for anode in store.anodes:
        for iface in anode.faces:
            face = store.faces[iface]
            for iplane in face.planes:
                plane = store.planes[iplane]
                for wip,wind in enumerate(plane.wires):
                    wire = store.wires[wind]
                    if wire.channel in channels:
                        bychan[wire.channel].append((wire, wip))

    fig, ax = plt.subplots(nrows=1, ncols=1)

    for ch,wws in sorted(bychan.items()):
        wws.sort(key=lambda ww: ww[0].segment)
        for wire, wip in wws:
            p1 = store.points[wire.tail]
            p2 = store.points[wire.head]
            width = wire.segment + 1
            ax.plot((p1.z/units.meter, p2.z/units.meter),
                    (p1.y/units.meter, p2.y/units.meter), linewidth = width)
            x = p2.z/units.meter
            y = p2.y/units.meter
            if x > 0:
                hal="left"
            else:
                hal="right"
            if labels:
                t='w:%d ch:%d\nident:%d seg:%d' % \
                   (wip, wire.channel, wire.ident, wire.segment)
                ax.text(x, y, t,
                        horizontalalignment=hal,
                        bbox=dict(facecolor='yellow', alpha=0.5, pad=10))
            ax.set_xlabel("Z [meter]")
            ax.set_ylabel("Y [meter]")
    fig.savefig(pdffile)

    

def allplanes(store, pdffile):
    '''
    Plot each plane of wires on a page of a PDF file.
    '''
    wire_step = 10                            # how many wires to skip

    from matplotlib.backends.backend_pdf import PdfPages

    # make some global pltos
    all_wire_x1 = list()
    all_wire_x2 = list()
    all_wire_z1 = list()
    all_wire_z2 = list()
    all_wire_anode = list()

    plane_colors=["blue","red","black"]

    with PdfPages(pdffile) as pdf:
        # make channel plots
        for anode in store.anodes:
            edge_z = list()
            edge_x = list()
            edge_n = list()
            edge_s = list()

            for iface in anode.faces:            
                face = store.faces[iface]

                for iplane in face.planes:
                    plane = store.planes[iplane]
                    wires_in_plane = [store.wires[wind] for wind in plane.wires]
                    wires = [w for w in wires_in_plane if w.segment == 0]
                    def pt(w): return store.points[w.head]
                    wires.sort(key=lambda a: pt(a).z)
                    def add_edge(w):
                        p = pt(w)
                        print (p,w.channel)
                        edge_z.append(p.z/units.m)
                        edge_x.append(p.x/units.m)
                        edge_n.append(w.channel)
                        edge_s.append('f%d p%d c%d wid%d' % (face.ident, plane.ident, w.channel,  w.ident))
                    add_edge(wires[0])
                    add_edge(wires[-1])

            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.scatter(edge_z, edge_x, s=1, c='red', marker='.')
            for i,(z,x,s) in enumerate(zip(edge_z, edge_x, edge_s)):
                hal = ["left","right"][i%2]
                ax.text(z, x, s, horizontalalignment=hal,
                        bbox=dict(facecolor='yellow', alpha=0.5, pad=1))
            for i in range(len(edge_n)//2):
                z = 0.5*(edge_z[2*i]+edge_z[2*i+1])
                n = 1+abs(edge_n[2*i] - edge_n[2*i+1])
                x = edge_x[2*i]
                ax.text(z, x, str(n), horizontalalignment='center',
                        bbox=dict(facecolor='yellow', alpha=0.5, pad=1))
                

            ax.set_title("Edge Channels AnodeID: %d" % (anode.ident))
            ax.set_xlabel("Z [meter]")
            ax.set_ylabel("X [meter]")
            pdf.savefig(fig)
            plt.close()

        for anode in store.anodes:
            seg_x1 = [list(),list(),list()]
            seg_x2 = [list(),list(),list()]
            seg_z1 = [list(),list(),list()]
            seg_z2 = [list(),list(),list()]
            seg_col = [list(),list(),list()]
            for iface in anode.faces:
                face = store.faces[iface]
                for iplane in face.planes:
                    plane = store.planes[iplane]
                    for wind in plane.wires:
                        wire = store.wires[wind]
                        p1 = store.points[wire.tail]
                        p2 = store.points[wire.head]
                        seg = wire.segment
                        seg_x1[seg].append(p1.x/units.meter)
                        seg_x2[seg].append(p2.x/units.meter)
                        seg_z1[seg].append(p1.z/units.meter)
                        seg_z2[seg].append(p2.z/units.meter)
                        seg_col[seg].append(plane_colors[iplane%(len(plane_colors))])
                        continue # wires
                    continue     # planes
                continue         # faces

            fig, axes = plt.subplots(nrows=3, ncols=1)
            for seg in range(3):
                ax = axes[seg]
                ax.scatter(seg_z2[seg], seg_x2[seg], c=seg_col[seg], s=1, marker='.')
                ax.set_title("AnodeID %d wires, seg %d, head (%d wires)" %
                             (anode.ident, seg, len(seg_col[seg])))
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

            fig, axes = plt.subplots(nrows=3, ncols=1)
            for seg in range(3):
                ax = axes[seg]
                ax.scatter(seg_z1[seg], seg_x1[seg], c=seg_col[seg], s=1, marker='.')
                ax.set_title("AnodeID %d wires, seg %d, tail (%d wires)" %
                             (anode.ident, seg, len(seg_col[seg])))
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
            continue            # anodes

        for anode in store.anodes:
            wire_x1 = list()
            wire_x2 = list()
            wire_z1 = list()
            wire_z2 = list()
            wire_anode = list()


            for iface in anode.faces:
                face = store.faces[iface]

                first_wires = list()
                
                for iplane in face.planes:
                    plane = store.planes[iplane]

                    print ("anodeID:%d faceID:%d planeID:%d" %
                           (anode.ident, face.ident, plane.ident))

                    first_wires.append(plane.wires[:2])

                    fig, ax = plt.subplots(nrows=1, ncols=1)
                    ax.set_aspect('equal','box')
                    for wind in plane.wires[::wire_step]:
                        wire = store.wires[wind]
                        p1 = store.points[wire.tail]
                        p2 = store.points[wire.head]
                        width = wire.segment + .1
                        ax.plot((p1.z/units.meter, p2.z/units.meter),
                                (p1.y/units.meter, p2.y/units.meter), linewidth = width)
                        wire_x1.append(p1.x/units.meter)
                        wire_z1.append(p1.z/units.meter)
                        wire_x2.append(p2.x/units.meter)
                        wire_z2.append(p2.z/units.meter)
                        wire_anode.append(anode.ident)

                    wirex = None
                    for wcount, wind in enumerate([plane.wires[0], plane.wires[-1]]):
                        wire = store.wires[wind]
                        print ("\twcount:%d wind:%d wident:%d chan:%d" % (wcount,wind,wire.ident,wire.channel))
                        p1 = store.points[wire.tail]
                        p2 = store.points[wire.head]
                        x = p2.z/units.meter
                        y = p2.y/units.meter
                        wirex = p2.x/units.meter
                        hal="center"
                        # if wcount == 1:
                        #    hal = "right"

                        t='%s wid:%d ch:%d' %(["beg","end"][wcount], wire.ident, wire.channel)
                        ax.text(x, y, t,
                                    horizontalalignment=hal,
                                    bbox=dict(facecolor='yellow', alpha=0.5, pad=10))

                    # if anode.ident==1 and face.ident==1:
                    #     for wcount, wind in enumerate(plane.wires):
                    #         wire = store.wires[wind]
                    #         if plane.ident==0 and wire.ident == 491 or \
                    #            plane.ident==1 and wire.ident == 514 or \
                    #            plane.ident==2 and wire.ident == 256:
                            
                    #                 p1 = store.points[wire.tail]
                    #                 p2 = store.points[wire.head]
                    #                 print("A1, F1, P%d, W: %s [%s -> %s]" %(plane.ident, wire, p1, p2))
                            


                    ax.set_xlabel("Z [meter]")
                    ax.set_ylabel("Y [meter]")
                    ax.set_title("AnodeID %d, FaceID %d, PlaneID %d every %dth wire, x=%.3fm" % \
                                 (anode.ident, face.ident, plane.ident, wire_step, wirex))
                    pdf.savefig(fig)
                    plt.close()
                    continue    # over planes

                # plot directions
                fig, axes = plt.subplots(nrows=2, ncols=2)
                for iplane,winds in enumerate(first_wires):
                    plane_color = "red green blue".split()[iplane]
                    w0 = store.wires[winds[0]]
                    h0 = numpy.asarray(store.points[w0.head])
                    t0 = numpy.asarray(store.points[w0.tail])
                    w1 = store.wires[winds[1]]
                    h1 = numpy.asarray(store.points[w1.head])
                    t1 = numpy.asarray(store.points[w1.tail])

                    c0 = 0.5*(h0+t0)
                    c1 = 0.5*(h1+t1)                    

                    # print ("A%d F%d c0=%s c1=%s" % (anode.ident, face.ident, c0, c1))
                    # print (winds[0], w0, winds[1], w1)
                    # print ("h0=%s t0=%s" % (h0, t0))
                    # print ("h1=%s t1=%s" % (h1, t1))

                    w = h0-t0   # wire direction
                    w = w/math.sqrt(numpy.sum(w*w))

                    r = c1-c0    # roughly in the pitch direction
                    r = r/math.sqrt(numpy.sum(r*r))

                    x = numpy.cross(w,r) # drift direction
                    x = x/math.sqrt(numpy.sum(x*x))
                    p = numpy.cross(x,w) # really pitch direction
                    
                    for ipt, pt in enumerate([x,w,p]):
                        axes[0,0].arrow(0,0, pt[2], pt[1], color=plane_color, linewidth=ipt+1) # 0
                        axes[0,1].arrow(0,0, pt[2], pt[0], color=plane_color, linewidth=ipt+1) # 1
                        axes[1,0].arrow(0,0, pt[0], pt[1], color=plane_color, linewidth=ipt+1) # 2

                    for a,t in zip(axes.flatten(),["Z vs Y","X vs Y","Z vs X","none"]):
                        a.set_aspect('equal')
                        a.set_title("%s anode: %d, face: %d" % (t, anode.ident, face.ident))
                        a.set_ylim(-1.1,1.1)
                        a.set_xlim(-1.1,1.1)

                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()

                continue        # over faces



            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.scatter(wire_z1, wire_x1,s=1, c=wire_anode, marker='.')
            ax.set_title("AnodeID %d wires, tail" % anode.ident)
            pdf.savefig(fig)
            plt.close()
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.scatter(wire_z2, wire_x2,s=1, c=wire_anode, marker='.')
            ax.set_title("AnodeID %d wires, head" % anode.ident)
            pdf.savefig(fig)
            plt.close()
            all_wire_x1 += wire_x1
            all_wire_z1 += wire_z1
            all_wire_x2 += wire_x2
            all_wire_z2 += wire_z2
            all_wire_anode += wire_anode
            
            continue            # over anodes
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.scatter(all_wire_z1, all_wire_x1,s=1, c=all_wire_anode,marker='.')
        ax.set_title("All wires, tail")
        pdf.savefig(fig)
        plt.close()
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.scatter(all_wire_z2, all_wire_x2,s=1, c=all_wire_anode,marker='.')
        ax.set_title("All wires, head")
        pdf.savefig(fig)
        plt.close()


def face_in_allplanes(store, iface=0, segments=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    face = store.faces[iface]
    for planeid in face.planes:
        plane = store.planes[planeid]
        for wind in plane.wires[::20]:
            wire = store.wires[wind]
            if segments and not wire.segment in segments:
                continue

            p1 = store.points[wire.tail]
            p2 = store.points[wire.head]

            width = wire.segment + 1
            ax.plot((p1.z, p2.z), (p1.y, p2.y), linewidth = width)

    return fig,ax

def allwires(store):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    iplane=0
    plane = store.planes[store.faces[0].planes[iplane]]
    wires = [store.wires[w] for w in plane.wires]
    nwires = len(wires)

    cmap = plt.get_cmap('seismic')
    colors = [cmap(i) for i in numpy.linspace(0, 1, nwires)]

    for iwire, wire in enumerate(wires):
        p1 = store.points[wire.tail]
        p2 = store.points[wire.head]

        color = colors[iwire]
        ax.plot((p1.z, p2.z), (p1.y, p2.y), color=color)

    return fig,ax





















def plot_rect(rect, color="black"):
    ax = plt.axes()
    ax.add_patch(mpatches.Rectangle(rect.ll, rect.width, rect.height,
                                        color=color, fill=False))
    ax.set_xlabel("APA-local Z")
    ax.set_ylabel("APA-local Y")
    ax.set_title("Looking in anti-drift direction")
    

def plot_polyline(pts):
    cmap = plt.get_cmap('seismic')
    npts = len(pts)
    colors = [cmap(i) for i in numpy.linspace(0, 1, npts)]
    for ind, (p1, p2) in enumerate(zip(pts[:-1], pts[1:])):
        x = numpy.asarray((p1.x, p2.x))
        y = numpy.asarray((p1.y, p2.y))
        plt.plot(x, y,  linewidth=ind+1)
    

def plotwires(wires):
    cmap = plt.get_cmap('seismic')
    nwires = len(wires)

    chans = [w[2] for w in wires]
    minchan = min(chans)
    maxchan = max(chans)
    nchans = maxchan - minchan + 1

    colors = [cmap(i) for i in numpy.linspace(0, 1, nchans)]
    for ind, one in enumerate(wires):
        pitch, side, ch, seg, p1, p2 = one
        linestyle = 'solid'
        if side < 0:
            linestyle = 'dashed'
        color = colors[ch-minchan]

        x = numpy.asarray((p1.x, p2.x))
        y = numpy.asarray((p1.y, p2.y))
        plt.plot(x, y, color=color, linewidth = seg+1, linestyle = linestyle)

def plot_wires_sparse(wires, indices, group_size=40):
    for ind in indices:
        plotwires([w for w in wires if w[2]%group_size == ind])


def plot_some():
    rect = Rectangle(6.0, 10.0)
    plt.clf()
    direc = Point(1,-1);
    for offset in numpy.linspace(.1, 6, 60):
        start = Point(-3.0 + offset, 5.0)
        ray = Ray(start, start+direc)
        pts = wrap_one(ray, rect)
        plot_polyline(pts)


        

def plot_wires(wobj, wire_filter=None):
    bbmin, bbmax = wobj.bounding_box
    xmin, xmax = bbmin[2],bbmax[2]
    ymin, ymax = bbmin[1],bbmax[1]
    dx = xmax-xmin
    dy = ymax-ymin
    wires = wobj.wires

    #print (xmin,ymin), (dx,dy)
    #print bbmin, bbmax

    wirenums = [w.wire for w in wires]
    minwire = min(wirenums)
    maxwire = max(wirenums)
    nwires = maxwire-minwire+1

    if wire_filter:
        wires = [w for w in wires if wire_filter(w)]
        print ("filter leaves %d wires" % len(wires))
    ax = plt.axes()
    ax.set_aspect('equal', 'box') #'datalim')
    ax.add_patch(mpatches.Rectangle((xmin, ymin), dx, dy,
                                    color="black", fill=False))

    cmap = plt.get_cmap('rainbow')        # seismic is bluewhitered

    colors = [cmap(i) for i in numpy.linspace(0, 1, nwires)]
    for ind, one in enumerate(wires):
        color = colors[one.wire-minwire]
        x = numpy.asarray((one.beg[2], one.end[2]))
        y = numpy.asarray((one.beg[1], one.end[1]))
        plt.plot(x, y, color=color)

    plt.plot([ xmin + 0.5*dx ], [ ymin + 0.5*dy ], "o")

    plt.axis([xmin,xmax,ymin,ymax])
    
