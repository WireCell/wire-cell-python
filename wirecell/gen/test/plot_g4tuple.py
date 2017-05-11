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

def plot_dndx(depos):
    dN = depos[:,4]
    dX = depos[:,5]
    dNdX = dN/dX
    print dNdX[0]
    h = numpy.histogram(dNdX, 150, (5.0e4,2.0e5))
    plt.clf(); plt.semilogy(h[1][:-1], h[0])
    plt.savefig("g4tuple-dndx.pdf")
    pass

def plot_nxz(depos):
    "Plot colz number as X vs Z (transverse)"
    x = depos[:,0]
    z = depos[:,2]
    n = depos[:,4]
    

    xedges = numpy.linspace(90, 120.0, 300)
    zedges = numpy.linspace(10, 70.0, 600)

    #H, xedg, zedg = numpy.histogram2d(z, x, bins=(xedges, zedges), weights=n)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Electrons per mm$^2$")
    #X, Z = numpy.meshgrid(xedges, zedges)
    #cax = ax.pcolormesh(X, Z, H)
    #cax = ax.imshow(H)#, interpolation='nearest', origin='low',
    #                   # extent=[xedg[0], xedg[-1], zedg[0], zedg[-1]])
    h = ax.hist2d(x,z,bins=(xedges, zedges), weights=n)
    ax.set_xlabel("X [cm]")
    ax.set_ylabel("Z [cm]")
    plt.colorbar(h[3], ax=ax)
    fig.savefig("g4tuple-nxz.pdf")


if '__main__' == __name__:
    import sys
    depos = load_depos(sys.argv[1])
    plot_dedx(depos);
    plot_dndx(depos);
    plot_nxz(depos);
