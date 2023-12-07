#!/usr/bin/env python
from warnings import warn

import numpy
import matplotlib.pyplot as plt
from wirecell.util.plottools import pages


from wirecell.util.wires import schema as wschema
from wirecell.util.wires import info as winfo
from wirecell.util.wires import array as warray
from wirecell.util.wires import persist as wpersist
from wirecell.util import detectors


def test_load_correct_transform():
    detname = "pdsp"
    store = wpersist.load(detname)

    assert isinstance(store, wschema.Store)
    assert isinstance(store.detectors[0], wschema.Detector)
    assert isinstance(store.wires[0], wschema.Wire)

    # get wiress from "first" plane, give more args to get others
    warr = warray.endpoints_from_schema(store)

    # most of our wire files are not actually correct.  This tries to correct
    # some problems.  Remake a data file with the wcwires program for more
    # corrections.
    wcor = warray.correct_endpoint_array(warr)

    if not numpy.all(wcor == warr):
        warn(f'{detname} wires file needed correction')

    wmean, pmean = warray.mean_wire_pitch(wcor)
    print(f'{pmean=}')
    R = warray.rotation(wmean, pmean)
    assert R.shape == (3,3)
    T = warray.translation(wcor)
    assert T.shape == (3,)    

    pmag = numpy.linalg.norm(pmean)
    print(f'pmag={pmag}')

    pdir = pmean / pmag
    wdir = wmean / numpy.linalg.norm(wmean)

    with pages("test_load_correct_transform.pdf") as out:
        # fixme: set aspect ratio

        plt.gca().set_aspect('equal')
        plt.title(f"{detname} corrected wires")
        plt.ylabel("Global Y")
        plt.xlabel("Global Z")
        sel = wcor[::10,:,:]
        for t,h in sel:
            x = t[2]            # z
            y = t[1]            # y
            dx = h[2]-t[2]
            dy = h[1]-t[1]
            plt.arrow(x,y,dx,dy, head_width=50)
        out.savefig()

        # demo using of rotation matrix
        plt.clf()
        plt.gca().set_aspect('equal')
        plt.ylabel("Global Y dir")
        plt.xlabel("Global Z dir")
        plt.title(f"wire plane coord rotation")
        plt.arrow(0,0, 1, 0, head_width=0.1)
        plt.arrow(0,0, 0, 1, head_width=0.1)
        for ind,axis in enumerate('xyz'):
            if not ind: continue
            _,dy,dx = R[ind]
            plt.arrow(0,0, dx, dy, head_width=0.1)
            plt.text(dx,dy, f'{axis}_wp')
        
        g = numpy.array([0.0, 1.0, 1.0])
        g /= numpy.linalg.norm(g)
        w = R@g
        plt.arrow(0,0, g[2], g[1], head_width=0, color="red")
        yw = w[1]*R[1]
        zw = w[2]*R[2]
        plt.scatter([yw[2],zw[2]], [yw[1], zw[1]], color="red")
        out.savefig()            
        
        # pick a points relative to the middle
        plt.clf()
        plt.gca().set_aspect('equal')
        plt.ylabel("Global Y")
        plt.xlabel("Global Z")
        plt.title(f"pitch offset from central wire centers")
        # center wire segment and midpoint
        mid_ind = wcor.shape[0]//2
        ct,ch = wcor[mid_ind,:,:]
        cm = 0.5*(ch+ct)
        ct2,ch2 = wcor[mid_ind+1,:,:]
        cm2 = 0.5*(ch2+ct2)
        
        # full, half pitch
        fp = pmag*pdir
        hp = 0.5*fp

        plt.arrow(cm[2], cm[1], wdir[2], wdir[1], head_width=0.1)
        plt.text(cm[2]+wdir[2], cm[1]+wdir[1], f"wire {mid_ind}")
        nmore=-2
        plt.arrow(cm[2], cm[1], nmore*wdir[2], nmore*wdir[1], head_width=0, linestyle='-', color='gray')

        nmore=10
        plt.arrow(cm2[2], cm2[1], nmore*wdir[2], nmore*wdir[1], head_width=0, linestyle='-', color='gray')
        plt.arrow(cm2[2], cm2[1], wdir[2], wdir[1], head_width=0.1)
        plt.text(cm2[2]+wdir[2], cm2[1]+wdir[1], f"wire {mid_ind+1}")

        plt.arrow(cm[2], cm[1], fp[2], fp[1], head_width=0,color="green")
        plt.text(cm[2]+fp[2], cm[1]+fp[1], "full pitch")

        plt.arrow(cm[2], cm[1], hp[2], hp[1], head_width=0.1,color="red")
        plt.text(cm[2]+hp[2], cm[1]+hp[1], "half pitch")
        out.savefig()            
        
