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
