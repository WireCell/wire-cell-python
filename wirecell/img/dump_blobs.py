#!/usr/bin/env python3
'''
Dump out signatures of blobs for debugging
signature: [tmin, tmax, umin, umax, vmin, vmax, wmin, wmax]
'''
from wirecell import units
import matplotlib.pyplot as plt
import numpy

def _signature(gr, node, tick=500):
    sig = []
    id2name = {1:'u', 2:'v', 4:'w'}
    channels = dict()
    chan_status = dict()
    for id in id2name:
        channels[id] = []
        chan_status[id] = []
    for node in gr.neighbors(node):
        ndata = gr.nodes[node]
        if ndata['code'] == 's':
            # print(ndata)
            tmin = ndata['start']//tick
            tmax = tmin + ndata['span']//tick
            sig.append(tmin)
            sig.append(tmax)
            signal = dict()
            for key in ndata['signal']:
                signal[int(key)] = ndata['signal'][key]
            for key in sorted(signal):
                print(key, ': ', signal[key]['val'])
    for node in gr.neighbors(node):
        ndata = gr.nodes[node]
        if ndata['code'] == 'w':
            print(ndata)
            # chid: global; index: per-plane
            channels[ndata['wpid']].append(ndata['index'])
            # chid = ndata['chid']
            # val = signal[chid]['val']
            # chan_status[ndata['wpid']].append(val)
    for wpid in channels:
        print(wpid, channels[wpid], chan_status[wpid])
        min = numpy.min(channels[wpid])
        max = numpy.max(channels[wpid])
        sig.append(min)
        sig.append(max)
    return sig

def _sort(arr,cols=[2,3,4,5,6,7]):
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
        if sig[0] == 0:
            sigs.append(sig)
    sigs = numpy.array(sigs)
    sigs = _sort(sigs)
    print(sigs.shape)
    print(sigs[0:100,:])
    numpy.save(out_file, sigs)