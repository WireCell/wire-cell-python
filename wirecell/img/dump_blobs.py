#!/usr/bin/env python3
'''
Dump out signatures of blobs for debugging
signature: [tmin, tmax, umin, umax, vmin, vmax, wmin, wmax]
'''
from wirecell import units
import matplotlib.pyplot as plt
import numpy
import math

def _signature(gr, bnode, tick=500):
    sig = []
    id2name = {1:'u', 2:'v', 4:'w'}
    chan_index = dict()
    chan_status = dict()
    signal = dict()
    for id in id2name:
        chan_index[id] = []
        chan_status[id] = []
    for node in gr.neighbors(bnode):
        ndata = gr.nodes[node]
        if ndata['code'] == 's':
            # print(ndata)
            tmin = ndata['start']//tick
            tmax = tmin + ndata['span']//tick
            sig.append(tmin)
            sig.append(tmax)
            for key in ndata['signal']:
                signal[int(key)] = ndata['signal'][key]
    for node in gr.neighbors(bnode):
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
            # status = -1
            # # print(f'val = {val}')
            # if math.isclose(val, 0.1,rel_tol=1e-6):
            #     status = 1
            # elif math.isclose(val, 0.2,rel_tol=1e-6):
            #     status = 0
            chan_status[wpid].append(val)
    chan_offset = {1:0, 2:2400, 4: 4800}
    for wpid in chan_index:
        # print(wpid, chan_index[wpid])
        if len(chan_index[wpid]) == 0:
            # print(gr.nodes[bnode])
            # for node in gr.neighbors(bnode):
            #     ndata = gr.nodes[node]
            #     print(ndata)
            # exit(-1)
            return None
        min = numpy.min(chan_index[wpid]) + chan_offset[wpid]
        max = numpy.max(chan_index[wpid]) + chan_offset[wpid]
        sig.append(min)
        sig.append(max)
    for wpid in chan_status:
        sig.append(int(sum(chan_status[wpid])))
    # print(sig)
    return sig

def _sort(arr):
    ind = numpy.lexsort((arr[:,7],arr[:,6],arr[:,5],arr[:,4],arr[:,3],arr[:,2],arr[:,1],arr[:,0]))
    arr = numpy.array([arr[i] for i in ind])
    return arr

def dump_blobs(gr, out_file):
    sigs = []
    count = 0
    for node, ndata in gr.nodes.data():
        if ndata['code'] != 'b':
            continue;
        sig = _signature(gr, node)
        # print(type(node))
        # print(sig)
        # exit()
        if sig is not None:
            sigs.append(sig)
        else:
            count += 1
    print('#0-blobs:', count)
    sigs = numpy.array(sigs)
    # sigs = sigs[sigs[:,0]==0,:]
    sigs = sigs[sigs[:,0]<40,:]
    # sigs = sigs[sigs[:,6]<5000,:]
    sigs = _sort(sigs)
    print(sigs.shape)
    # for i in range(min([sigs.shape[0], 20])):
    for i in range(sigs.shape[0]):
        # print(i, sigs[i,:])
        print(sigs[i,0:2],
            sigs[i,2], ':', sigs[i,3]+1, ',',
            sigs[i,4], ':', sigs[i,5]+1, ','
            ,sigs[i,6], ':', sigs[i,7]+1
            ,sigs[i,8:]
            )
    if out_file is not None:
        numpy.save(out_file, sigs)