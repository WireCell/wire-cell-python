#!/usr/bin/env python

import numpy
import matplotlib.pyplot as plt

import json
import bz2

#          0123456
depo_keys="xyzqtsn"

def load_depos(filename):
    if filename.endswith(".bz2"):
        fp = bz2.BZ2File(filename)
    else:
        fp = open(filename);
    jdat = json.loads(fp.read())
    depos = list()
    for jdepo in jdat["depos"]:
        depo = tuple([jdepo[key] for key in depo_keys])
        if depo[5] == 0.0:
            continue
        depos.append(depo)
    return numpy.asarray(depos)

def plot_dedx(depos):
    dE = depos[:,3]
    dX = depos[:,5]
    dEdX = dE/dX
    h = numpy.histogram(dEdX, 1000, (0,10))
    plt.clf(); plt.plot(h[1][:-1], h[0])
    plt.savefig("g4tuple-dedx.pdf")
    pass




if '__main__' == __name__:
    import sys
    depos = load_depos(sys.argv[1])
    plot_dedx(depos);
