#!/usr/bin/env python

from wirecell import units

import numpy
import matplotlib.pyplot as plt

import json
import bz2


#        0123456
columns="xyzqtsn"

def todict(depos):
    'Return a dictionary of arrays instead of a 2D array.'
    ret = dict()
    for ind, letter in enumerate(columns):
        ret[letter] = depos[:,ind]
    return ret

def remove_zero_steps(depos):
    '''
    For some reason sometimes zero steps are taken.  This removes them
    '''
    ret = list()
    for depo in depos:
        if depo[5] == 0.0:
            continue
        ret.append(depo)
    return numpy.asarray(ret)


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


def apply_units(depos, distance_unit, time_unit, energy_unit, step_unit=None, electrons_unit="1.0"):
    'Apply units to a deposition array, return a new one'

    print "dtese=", distance_unit, time_unit, energy_unit, step_unit, electrons_unit,

    depos = numpy.copy(depos)

    dunit = eval(distance_unit, units.__dict__)
    tunit = eval(time_unit, units.__dict__)
    eunit = eval(energy_unit, units.__dict__)
    if step_unit is None:
        sunit = dunit
    else:
        sunit = eval(step_unit, units.__dict__)
    nunit = eval(electrons_unit, units.__dict__)

    theunits = [dunit]*3 + [eunit] + [tunit] + [sunit] + [nunit]
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


#          0123456
# columns="xyzqtsn"
def plot_hist(h, xlab, output):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlab)
    ax.semilogy(h[1][:-1], h[0])
    fig.savefig(output)
    

def plot_s(depos, output):
    depos = todict(depos)
    s = depos["s"]/units.cm
    h = numpy.histogram(s, 1000, (0, 0.05))
    plot_hist(h, "step [cm]", output)
def plot_q(depos, output):
    depos = todict(depos)
    q = depos["q"]/units.MeV
    h = numpy.histogram(q, 1000, (0, 0.2))
    plot_hist(h, "deposit [MeV]", output)
def plot_n(depos, output):
    depos = todict(depos)
    n = depos["n"]
    h = numpy.histogram(n, 1000, (0, 10000))
    plot_hist(h, "number [electrons]", output)
def plot_deds(depos, output):
    depos = todict(remove_zero_steps(depos))
    q = depos["q"]/units.MeV
    s = depos["s"]/units.cm
    h = numpy.histogram(q/s, 1000, (0,10))
    plot_hist(h, "q/s [MeV/cm]", output)
def plot_dnds(depos, output):
    depos = todict(remove_zero_steps(depos))
    s = depos["s"]/units.cm
    n = depos["n"]
    h = numpy.histogram(n/s, 500, (0,5.0e5))
    plot_hist(h, "n/s [#/cm]", output)


def plot_xz_weighted(x, z, w, title, output):
    x = x/units.mm
    z = z/units.mm
    xmm = (numpy.min(x), numpy.max(x))
    zmm = (numpy.min(z), numpy.max(z))

    # assuming data is in WCT SoU we want mm bins 
    nx = int((xmm[1] - xmm[0])/units.mm)
    nz = int((zmm[1] - zmm[0])/units.mm)
    print ("%d x %d pixels: " % (nx, nz))

    xedges = numpy.linspace(xmm[0], xmm[1], nx)
    zedges = numpy.linspace(zmm[0], zmm[1], nz)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)

    h = ax.hist2d(x,z,bins=(xedges, zedges), weights=w)
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Z [mm]")
    plt.colorbar(h[3], ax=ax)
    fig.savefig(output)


def plot_qxz(depos, output):
    'Plot colz q as X vs Z'
    depos = todict(remove_zero_steps(depos))
    x = depos["x"]
    z = depos["z"]
    q = depos["q"]/units.MeV
    plot_xz_weighted(x, z, q, "q [MeV] per mm$^2$", output)


def plot_qsxz(depos, output):
    'Plot colz q/s as X vs Z'
    depos = todict(remove_zero_steps(depos))
    x = depos["x"]
    z = depos["z"]
    q = depos["q"]/units.MeV
    s = depos["s"]/units.cm
    plot_xz_weighted(x, z, q/s, "q/s [MeV/cm] per mm$^2$", output)
    

def plot_nxz(depos, output):
    "Plot colz number as X vs Z (transverse)"
    depos = todict(remove_zero_steps(depos))
    x = depos["x"]
    z = depos["z"]
    n = depos["n"]
    plot_xz_weighted(x, z, n, "number e- per mm$^2$", output)

def plot_nscat(depos, output):
    'Plot number as scatter plot'
    depos = todict(remove_zero_steps(depos))
    x = depos["x"]/units.mm
    z = depos["z"]/units.mm
    n = depos["n"]

    nmax = float(numpy.max(n))
    cmap = plt.get_cmap('seismic')

    sizes = [20.0*nv/nmax for nv in n]
    colors = [cmap(nv/nmax) for nv in n]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("electrons")

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Z [mm]")
    #plt.colorbar(h[3], ax=ax)
    ax.scatter(x, z, c=colors, edgecolor=colors, s=sizes)
    fig.savefig(output)
