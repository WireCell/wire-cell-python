        

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
    
