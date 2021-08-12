#!/usr/bin/env python3
'''
Util functions for the frame-split command
'''

import os
import json
import numpy

def save_one(path, aname, array, md=None, compress=True):
    '''
    Save named array to file at path
    '''
    dname = os.path.dirname(path)
    if dname and not os.path.exists(dname):
        os.makedirs(dname)
    tosave = {aname: array}
    if compress:
        numpy.savez_compressed(path, **tosave)
    else:
        numpy.savez(path, **tosave)        
    if not md:
        return
    jpath = path.replace(".npz",".json")
    with open(jpath,"w") as jp:
        jp.write(json.dumps(md, indent=4))

def offset_cols(arr, ncols=0, pad=0):
    '''
    Shift the content of arr by ncols, preserving shape and padding.

    If ncols is positive, shift content "right" (toward higher column
    number).
    '''
    if not ncols:
        return arr
    ret = numpy.zeros_like(arr) + pad
    wid = arr.shape[1]
    if abs(ncols) >= wid:
        return ret
    if ncols < 0:
        ret[:,:ncols] = arr[:,-ncols:] # shift down/left
    else:
        ret[:,ncols:] = arr[:,:-ncols] # shift up/right
    return ret

def rebin_cols(arr, ncols=0):
    '''
    Return array with each new column the sum of ncols columns.
    '''
    if not ncols:
        return arr
    ncols = arr.shape[1]//ncols
    return arr.reshape(arr.shape[0], ncols, -1).sum(axis=2)


def one_pdsp_array(planeid, frame, anodeid=0, tick_offset=0, rebin=0):
    '''
    Return one "split out" array from the frame.
    '''
    
    cranges = [0, 800, 1600, 2560]
    offset = 2560 * anodeid
    a = offset + cranges[planeid]
    b = offset + cranges[planeid+1]
    parr = frame[a:b, :]
    if tick_offset:
        parr = offset_cols(parr, tick_offset)
    if rebin:
        parr = rebin_cols(parr, rebin)
    return parr


def apa(frame, tag, index, tick_offset=0, rebin=0, anodeid=0, detector="protodune"):
    '''
    Split frame by plane as if it holds single APA array.

    Return list of (array, metadata) tuples
    '''
    ret = list()

    for planeid in range(3):
        md = dict(planeid=planeid, anodeid=anodeid,
                  tick_offset=tick_offset, rebin=rebin,
                  tag=tag, index=index, planeletter="UVW"[planeid],
                  detector="protodune")
        parr = one_pdsp_array(planeid, frame, anodeid,
                              tick_offset, rebin)
        ret.append((parr, md))
    return ret


def protodune(frame, tag, index, tick_offset=0, rebin=0):
    '''
    Split frame by plane as if it holds multi-APA array.

    Return list of (array, metadata) tuples
    '''
    ret = list()

    for anodeid in range(6):
        gots = apa(frame, tag, index, tick_offset, rebin, anodeid)
        ret += gots

    return ret


def guess_splitter(nchan):
    '''
    Return a splitter function based on number of channels.  

    See wirecell-util frame-split for details.
    '''
    if nchan == 2560:
        return apa
    if nchan == 15360:
        return protodune
    raise ValueError(f"Unknown splitter of {nchan} channels")


