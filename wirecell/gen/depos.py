#!/usr/bin/env python

from wirecell import units

import numpy
import matplotlib.pyplot as plt

import json
import bz2


#        0123456
columns="xyzqtsn"

def load(filename, jpath="depos"):
    '''
    Load a CWT JSON depo file and return numpy array of depo info.
    '''

    if filename.endswith(".json"):
        fopen = open
    elif filename.endswith(".bz2"):
        fopen = bz2.BZ2File
    else:
        return None

    with fopen(filename) as fp:
        jdat = json.loads(fp.read())

    depos = list()
    for jdepo in jdat["depos"]:
        depo = tuple([jdepo.get(key,0.0) for key in columns])
        depos.append(depo)
    return numpy.asarray(depos)


def apply_units(depos, distance_unit, time_unit, energy_unit, electrons_unit="1.0"):
    'Apply units to a deposition array, return a new one'

    depos = numpy.copy(depos)

    dunit = eval(distance_unit, units.__dict__)
    tunit = eval(time_unit, units.__dict__)
    eunit = eval(energy_unit, units.__dict__)
    nunit = eval(electrons_unit, units.__dict__)

    theunits = [dunit]*3 + [eunit] + [tunit] + [dunit] + [nunit]
    for ind,unit in enumerate(theunits):
        depos[:,ind] *= unit
    return depos



def dump(output_file, depos, jpath="depos"):
    '''
    Save a deposition array to JSON file 
    '''
    jlist = list()
    for depo in depos:
        jdepo = {c:v for c,v in zip(columns, depo)}
        jlist.append(jdepo)
    out = {jpath: jlist}

    # indent for readability.  If bz2 is used, there is essentially no change
    # in file size between no indentation and indent=4.  Plain JSON inflates by
    # about 2x.  bz2 is 5-10x smaller than plain JSON.
    text = json.dumps(out, indent=4) 

    if output_file.endswith(".json"):
        fopen = open
    elif output_file.endswith(".json.bz2"):
        fopen = bz2.BZ2File
    else:
        raise IOError('Unknown file extension: "%s"' % filename)
    with fopen(output_file, 'w') as fp:
        fp.write(text)
    return
    



def move(depos, offset):
    '''
    Return new set of depos all moved by given vector offset.
    '''
    offset = numpy.asarray(offset)
    depos = numpy.copy(depos)
    depos[:,0:3] += offset
    return depos
            

def center(depos, point):
    '''
    Shift depositions so that they are centered on the given point
    '''
    point = numpy.asarray(point)
    tot = numpy.asarray([0.0, 0.0, 0.0])
    for depo in depos:
        tot += depo[:3]
    n = len(depos)
    offset = point - tot/n
    return move(depos, offset)



def plot_dedx(depos, output):
    dE = depos[:,3]
    dX = depos[:,5]
    dEdX = dE/dX
    h = numpy.histogram(dEdX, 1000, (0,10))
    plt.clf(); plt.plot(h[1][:-1], h[0])
    plt.savefig(output)
    pass

def plot_dndx(depos, output):
    dN = depos[:,4]
    dX = depos[:,5]
    dNdX = dN/dX
    print dNdX[0]
    h = numpy.histogram(dNdX, 150, (5.0e4,2.0e5))
    plt.clf(); plt.semilogy(h[1][:-1], h[0])
    plt.savefig(output)
    pass

def plot_nxz(depos, output):
    "Plot colz number as X vs Z (transverse)"
    x = depos[:,0]
    z = depos[:,2]
    n = depos[:,4]
    
    
    xmm = (numpy.min(x), numpy.max(x))
    zmm = (numpy.min(z), numpy.max(z))

    # assuming data is in WCT SoU we want mm bins 
    nx = int((xmm[1] - xmm[0])/units.mm)
    nz = int((zmm[1] - zmm[0])/units.mm)
    print ("%d x %d pixels: " % (nx, nz))

    xedges = numpy.linspace(xmm[0], xmm[1], nx)
    zedges = numpy.linspace(zmm[0], zmm[1], nz)

    #H, xedg, zedg = numpy.histogram2d(z, x, bins=(xedges, zedges), weights=n)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Electrons per mm$^2$")
    #X, Z = numpy.meshgrid(xedges, zedges)
    #cax = ax.pcolormesh(X, Z, H)
    #cax = ax.imshow(H)#, interpolation='nearest', origin='low',
    #                   # extent=[xedg[0], xedg[-1], zedg[0], zedg[-1]])
    h = ax.hist2d(x,z,bins=(xedges, zedges), weights=n)
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Z [mm]")
    plt.colorbar(h[3], ax=ax)
    fig.savefig(output)


