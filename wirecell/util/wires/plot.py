import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy

def plot_polyline(pts):
    cmap = plt.get_cmap('seismic')
    npts = len(pts)
    colors = [cmap(i) for i in numpy.linspace(0, 1, npts)]
    for ind, (p1, p2) in enumerate(zip(pts[:-1], pts[1:])):
        x = numpy.asarray((p1.x, p2.x))
        y = numpy.asarray((p1.y, p2.y))
        plt.plot(x, y,  linewidth=ind+1)
    


def oneplane(store, iplane, iface=0, segments=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    face = store.faces[iface]
    planeid = face.planes[iplane]
    plane = store.planes[planeid]

    print segments
    for wind in plane.wires[::10]:
        wire = store.wires[wind]
        if segments and not wire.segment in segments:
            continue

        p1 = store.points[wire.tail]
        p2 = store.points[wire.head]
        width = wire.segment + 1
        ax.plot((p1.z, p2.z), (p1.y, p2.y), linewidth = width)
    return fig,ax

def allplanes(store, iface=0, segments=None):
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

    print (xmin,ymin), (dx,dy)
    print bbmin, bbmax

    wirenums = [w.wire for w in wires]
    minwire = min(wirenums)
    maxwire = max(wirenums)
    nwires = maxwire-minwire+1

    if wire_filter:
        wires = [w for w in wires if wire_filter(w)]
        print "filter leaves %d wires" % len(wires)
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
    
