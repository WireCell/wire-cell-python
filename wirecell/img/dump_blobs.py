#!/usr/bin/env python3
'''
Dump out signatures of blobs for debugging
signature: [tmin, tmax, umin, umax, vmin, vmax, wmin, wmax]
'''
from wirecell import units
import matplotlib.pyplot as plt
import numpy
import math

def _signature(gr, node, tick=500):
    sig = []
    id2name = {1:'u', 2:'v', 4:'w'}
    chan_index = dict()
    chan_status = dict()
    signal = dict()
    for id in id2name:
        chan_index[id] = []
        chan_status[id] = []
    for node in gr.neighbors(node):
        ndata = gr.nodes[node]
        if ndata['code'] == 's':
            # print(ndata)
            tmin = ndata['start']//tick
            tmax = tmin + ndata['span']//tick
            sig.append(tmin)
            sig.append(tmax)
            for key in ndata['signal']:
                signal[int(key)] = ndata['signal'][key]
    for node in gr.neighbors(node):
        ndata = gr.nodes[node]
        if ndata['code'] == 'w':
            # print(ndata)
            # chid: global; index: per-plane
            chid = ndata['chid']
            wpid = ndata['wpid']
            index = ndata['index']
            chan_index[wpid].append(index)
            if chid in signal:
                val = signal[chid]['val']
            else:
                val = -1
                # for key in sorted(signal):
                #     print(key, ': ', signal[key]['val'])
                # raise RuntimeError(f'{chid} not in signal')
            # 0.1: dummy; 0.2: masked
            status = -1
            # print(f'val = {val}')
            if math.isclose(val, 0.1,rel_tol=1e-6):
                status = 1
            elif math.isclose(val, 0.2,rel_tol=1e-6):
                status = 0
            chan_status[wpid].append(status)
    for wpid in chan_index:
        # FIXME why this can be 0?
        if len(chan_index[wpid]) == 0:
            # print(wpid, chan_index[wpid])
            sig.append(-1)
            sig.append(-1)
            continue
        min = numpy.min(chan_index[wpid])
        max = numpy.max(chan_index[wpid])
        sig.append(min)
        sig.append(max)
    for wpid in chan_status:
        # FIXME why this can be 0?
        if len(chan_status[wpid]) == 0:
            # print(wpid, chan_status[wpid])
            sig.append(-1)
            continue
        min = numpy.min(chan_status[wpid])
        max = numpy.max(chan_status[wpid])
        if min != max and min != -1:
            raise ValueError(f'min = {min}, max = {max}')
        sig.append(min)
    # print(sig)
    return sig

def _sort(arr):
    ind = numpy.lexsort((arr[:,7],arr[:,6],arr[:,5],arr[:,4],arr[:,3],arr[:,2]))
    arr = numpy.array([arr[i] for i in ind])
    return arr

def dump_blobs(gr, out_file):
    sigs = []
    for node, ndata in gr.nodes.data():
        if ndata['code'] != 'b':
            continue;
        sig = _signature(gr, node)
        # print(sig)
        # exit()
        sigs.append(sig)
    sigs = numpy.array(sigs)
    sigs = sigs[sigs[:,0]==0,:]
    sigs = _sort(sigs)
    print(sigs.shape)
    print(sigs[0:20,:])
    numpy.save(out_file, sigs)