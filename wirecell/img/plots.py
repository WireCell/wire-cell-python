import numpy
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.ticker import  AutoMinorLocator
from matplotlib.colors import LogNorm

from collections import defaultdict
from wirecell import units
from wirecell.util.wires.schema import plane_face_apa;


class Hist2D(object):
    def __init__(self, nx, xmin, xmax, ny, ymin, ymax):
        self.arr = numpy.zeros((ny, nx))
        self.nx = nx
        self.rangex = (xmin, xmax)
        self.ny = ny
        self.rangey = (ymin, ymax)

    def xbin(self, x):
        xmin,xmax=self.rangex
        xrel = max(0, min(1.0, (x - xmin) / (xmax-xmin)))
        return int(xrel * self.nx);

    def ybin(self, y):
        ymin,ymax=self.rangey
        yrel = max(0, min(1.0, (y - ymin) / (ymax-ymin)))
        return int(yrel * self.ny);

    def fill(self, x, y, v):
        xi = self.xbin(x)
        yi = self.ybin(y)
        self.arr[yi, xi] += v

    def extent(self):
        return (self.rangex[0], self.rangex[1],
                self.rangey[1], self.rangey[0])
    def imshow(self, ax):
        return ax.imshow(self.arr, extent=self.extent())

    def like(self):
        return Hist2D(self.nx, self.rangex[0], self.rangex[1],
                      self.ny, self.rangey[0], self.rangey[1])

def activity(cm):
    '''
    Given a ClusterMap, return a figure showing activity
    '''
    channels = set()
    slices = set()
    for snode in cm.nodes_oftype('s'):
        sdat = cm.gr.nodes[snode]
        for c in sdat['activity'].keys():
            channels.add(int(c))
        slices.add(sdat['ident'])
    
    cmin = min(channels);
    cmax = max(channels);
    smin = min(slices);
    smax = max(slices);
    print ("activity: c:[%d,%d], s:[%d,%d]" % (cmin, cmax, smin, smax))
    
    hist = Hist2D(smax-smin+1, smin, smax+1,
                  cmax-cmin+1, cmin, cmax+1)
    for snode in cm.nodes_oftype('s'):
        sdat = cm.gr.nodes[snode]
        si = sdat['ident']
        for c,v in sdat['activity'].items():
            ci = int(c)
            hist.fill(si+.1, ci+.1, v)

    return hist
    # fig,ax = plt.subplots(nrows=1, ncols=1)
    # im = hist.imshow(ax)
    # plt.colorbar(im, ax=ax)
    # return fig,ax,hist

def blobs(cm, hist):
    for bnode in cm.nodes_oftype('b'):
        bdat = cm.gr.nodes[bnode]
        for wnode in cm.gr[bnode]:
            wdat = cm.gr.nodes[wnode]
            if wdat['code'] != 'w':
                continue;
            
            hist.fill(bdat['sliceid']+0.1, wdat['chid']+.1, 1)
    return hist
    # fig,ax = plt.subplots(nrows=1, ncols=1)
    # im = hist.imshow(ax)
    # plt.colorbar(im, ax=ax)
    # return fig,ax,hist
    

def mask_blobs(a, b, sel=lambda a: a < 1, extent=None):
    '''
    Plot the activity with blobs masked out
    '''
    cmap = plt.get_cmap('gist_rainbow')
    #cmap = plt.cm.gist_rainbow
    cmap.set_bad(color='black')
    arr = numpy.ma.masked_where(sel(b), a)
    fig,ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(8.5,11.0)
    im = ax.imshow(arr, cmap=cmap, interpolation='none', extent=extent)
    minorLocator = AutoMinorLocator()
    ax.yaxis.set_minor_locator(minorLocator)
    ax.tick_params(which="both", width=1)
    ax.tick_params(which="major", length=7)
    ax.tick_params(which="minor", length=3)
    plt.colorbar(im, ax=ax)
    return fig, ax

def wire_blob_slice(cm, sliceid):
    '''
    Plot slice information as criss crossing wires and blob regions.
    '''
    from . import converter

    snodes = cm.find('s', ident=sliceid)
    if len(snodes) != 1:
        print ("Unexpected number of slices with ID: %d, found %d" % (sliceid, len(snodes)))
        return
    snode = snodes[0]
    by_face = defaultdict(list)
    sdata = cm.gr.nodes[snode]
    for cdat,cval in sdata["activity"].items():
        chid = int(cdat)
        cnode = cm.channel(chid)
        cdata = cm.gr.nodes[cnode]
        wnodes = cm.neighbors_oftype(cnode, 'w')
        if not wnodes:
            print("No wires for channel %d" % chid)
        for wnode in wnodes:
            wdat = cm.gr.nodes[wnode]
            wpid = wdat['wpid']
            p,f,a = plane_face_apa(wpid)
            by_face[f].append((cval, wdat))
    
    blob_xs_byface = defaultdict(list)
    blob_ys_byface = defaultdict(list)
    blob_cs_byface = defaultdict(list)
    for bnode in cm.neighbors_oftype(snode, 'b'):
        bdata = cm.gr.nodes[bnode]
        cx = list()
        cy = list()
        cpoints = converter.orderpoints(bdata['corners'])
        for cp in cpoints + [cpoints[0]]:
            cx.append(cp[2]/units.m)
            cy.append(cp[1]/units.m)
        faceid = bdata['faceid']
        blob_xs_byface[faceid].append(cx)
        blob_ys_byface[faceid].append(cy)
        blob_cs_byface[faceid].append(bdata['value'])


    cmap = plt.get_cmap('gist_rainbow')
    linewidth=0.1
    fig,axes = plt.subplots(nrows=1, ncols=len(by_face))



    for ind, (faceid, wdats) in enumerate(sorted(by_face.items())):
        ax = axes[ind]
        ax.set_title("face %d" % faceid)
        ax.set_xlabel("Z [m]")
        ax.set_ylabel("Y [m]")

        xs = list()
        ys = list()
        cs = list()
        for cval, wdat in wdats:
            cs.append(cval)
            h = wdat['head']
            t = wdat['tail']
            c = [0.5*(h[i]+t[i]) for i in range(3)]

            p,f,a = plane_face_apa(wdat['wpid'])
            wid = wdat['ident']
            wip = wdat['index']

            chid = wdat['chid']
            toffset = (chid%5) * 0.2

            if p == 4:      # collection
                if wdat['seg'] == 0:
                    ax.text(t[2]/units.m, t[1]/units.m + toffset, "C%d" %chid, fontsize=0.2, rotation=90, va='top')

                ax.text(c[2]/units.m, c[1]/units.m-toffset, "P%d WID%d WIP%d" %(p,wid,wip), fontsize=0.2, rotation=90, va='top')
                ax.plot([c[2]/units.m,c[2]/units.m], [c[1]/units.m, c[1]/units.m-toffset],
                        color='black', linewidth = 0.1, alpha=0.5)

            else:
                if wdat['seg'] == 0:
                    ax.text(h[2]/units.m, h[1]/units.m - toffset, "C%d" %chid, fontsize=0.2, rotation=90, va='bottom')
                    ax.plot([h[2]/units.m,h[2]/units.m], [h[1]/units.m - toffset, h[1]/units.m],
                            color='black', linewidth = 0.1, alpha=0.5)

                ax.text(c[2]/units.m+toffset, c[1]/units.m, "P%d WID%d WIP%d" %(p,wid,wip), fontsize=0.2, rotation=0, va='top')
                ax.plot([c[2]/units.m+toffset,c[2]/units.m], [c[1]/units.m, c[1]/units.m],
                        color='black', linewidth = 0.1, alpha=0.5)

            xs.append([t[2]/units.m, h[2]/units.m])
            ys.append([t[1]/units.m, h[1]/units.m])


        segments = [numpy.column_stack([x,y]) for x,y in zip(xs, ys)]
        lc = LineCollection(segments, cmap=cmap, linewidth=linewidth, alpha=0.5, norm=LogNorm())
        lc.set_array(numpy.asarray(cs))
        ax.add_collection(lc)
        ax.autoscale()
        fig.colorbar(lc, ax=ax)

        segments = [numpy.column_stack([x,y]) for x,y in zip(blob_xs_byface[faceid], blob_ys_byface[faceid])]
        lc = LineCollection(segments, linewidth=2*linewidth, alpha=0.5)
        lc.set_array(numpy.asarray(blob_cs_byface[faceid]))
        ax.add_collection(lc)


    # norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
    # cb = mpl.colorbar.ColorbarBase(axes[-1], cmap=cmap, norm=norm)
    # cb.set_label("signal charge")
    return fig, axes

