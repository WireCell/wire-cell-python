#!/usr/bin/env python3
'''
Helpers for plot-blobs main command

Exposed functions take a graph return a figure:

  func_<name>(gr) -> fig

'''
from wirecell import units
import matplotlib.pyplot as plt
import numpy

from .converter import get_slice

def subplots(nrows=1, ncols=1):
    return plt.subplots(nrows, ncols, tight_layout=True)

def _plot_coord(gr, index, label, unit):
    vals = list()
    for node, ndata in gr.nodes.data():
        if ndata['code'] != 'b':
            continue;
        v = ndata['corners'][0][index]
        vals.append(v)

    fig, ax = subplots()
    ax.hist(vals/unit, bins=1000)
    ax.set_label(label)
    gname = getattr(gr, "name", None)
    if gname:
        gname = f' ({gname})'
    letter = "xyz"[posindex]
    ax.set_title(f'Blob {letter.upper()} {gname}')
    return fig

def plot_x(gr): return _plot_coord(gr, 0, 'X [cm]', units.cm)
def plot_y(gr): return _plot_coord(gr, 1, 'Y [cm]', units.cm)
def plot_z(gr): return _plot_coord(gr, 2, 'Z [cm]', units.cm)

def plot_t(gr):
    '''
    Histogram blob times
    '''
    times = list()
    for node, ndata in gr.nodes.data():
        if ndata['code'] != 'b':
            continue;
        snode = get_slice(gr, node)
        sdat = gr.nodes[snode]
        times.append(sdat['start'])
    fig, ax = subplots()
    ax.hist(numpy.array(times)/units.us, bins=1000)
    ax.set_label('T [us]')
    gname = getattr(gr, "name", None)
    if gname:
        gname = f' ({gname})'
    ax.set_title(f'Blob T {gname}')
    return fig

def _plot_tN(gr, posindex):
    letter = "xyz"[posindex]

    time = list()
    ypos = list()
    for node, bdat in gr.nodes.data():
        if bdat['code'] != 'b':
            continue;
        snode = get_slice(gr, node)
        sdat = gr.nodes[snode]
        time.append(sdat['start'])
        y = numpy.sum(numpy.array(bdat['corners'])[:,posindex]) / len(bdat['corners'])
        ypos.append(y)
    fig, ax = subplots()
    ax.scatter(numpy.array(time)/units.us, numpy.array(ypos)/units.mm)
    gname = getattr(gr, "name", None)
    if gname:
        gname = f' ({gname})'
    ax.set_title(f'Blob <{letter.upper()}> vs T {gname}')
    ax.set_xlabel('t [us]')
    ax.set_ylabel(f'{letter} [mm]')
    return fig

def plot_tx(gr): return _plot_tN(gr, 0)
def plot_ty(gr): return _plot_tN(gr, 1)
def plot_tz(gr): return _plot_tN(gr, 2)

def plot_views(gr):
    '''
    Project blob charge onto each view and along slices
    '''
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection

    # eg: (v vs u, t vs u)
    fig, axes = subplots(2,3)

    blobs = [d for n,d in gr.nodes.data() if d['code']=='b']
    if len(blobs) == 0:
        raise ValueError('graph with no blobs')

    # (N,)
    qall = numpy.array([b['value'] for b in blobs])
    ind = qall>0
    q = qall[ind]
    if q.size == 0:
        raise ValueError("no non-negative blob charges")
    t = (numpy.array([b['start'] for b in blobs])/units.ms)[ind]
    dt = (numpy.array([b['span'] for b in blobs])/units.ms)[ind]
    wb = numpy.array([b['bounds'] for b in blobs])[ind,:,:]

    print (numpy.min(t), numpy.mean(t), numpy.max(t))
    print (numpy.min(dt), numpy.mean(dt), numpy.max(dt))

    # (N, 3, 2)

    alpha = 0.25
    cscale = 5.0
    
    for p1,p2 in zip([0,1,2], [1,2,0]):
        # (N,2) min/max wire in plane
        wb1 = wb[:,p1,:]
        wb2 = wb[:,p2,:]

        points = numpy.vstack((wb1[:,0], wb2[:,0])).T
        widths = wb1[:,1] - wb1[:,0]
        heights = wb2[:,1] - wb2[:,0]
        qdens = q/(widths*heights)
        qmean = numpy.mean(qdens)
        tdens = q/(widths*dt)
        tmean = numpy.mean(tdens)

        print(points.shape, widths.shape, heights.shape, qdens.shape, t.shape, dt.shape)

        r1 = [Rectangle(*args) for args in zip(points, widths, heights)]
        pc1 = PatchCollection(r1, alpha=alpha, cmap='viridis')
        pc1.set_array(qdens)
        pc1.set_clim([qmean/cscale, qmean*cscale])
        
        tpts = numpy.vstack((wb1[:,0], t)).T
        r2 = [Rectangle(*args) for args in zip(tpts, widths, dt)]
        pc2 = PatchCollection(r2, alpha=alpha, cmap='viridis')
        pc2.set_array(tdens)
        pc2.set_clim([tmean/cscale, tmean*cscale])

        ax1,ax2 = axes[:,p1]
        l1 = "UVW"[p1]
        l2 = "UVW"[p2]

        # 2-plane wire projection
        ax1.add_collection(pc1)
        ax1.set_xlim(numpy.min(points[:,0]), numpy.max(points[:,0]+widths))
        ax1.set_ylim(numpy.min(points[:,1]), numpy.max(points[:,1]+heights))
        ax1.set_xlabel(l1)
        ax1.set_ylabel(l2)
        fig.colorbar(pc1, ax=ax1)

        # time vs wires of one plane
        ax2.add_collection(pc2)
        ax2.set_xlim(numpy.min(tpts[:,0]), numpy.max(tpts[:,0]+widths))
        ax2.set_ylim(numpy.min(tpts[:,1]), numpy.max(tpts[:,1]+dt))
        ax2.set_xlabel(l1)
        ax2.set_ylabel("T [ms]")
        fig.colorbar(pc2, ax=ax2)

    return fig

