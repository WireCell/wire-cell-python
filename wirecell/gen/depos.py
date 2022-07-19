#!/usr/bin/env python

from wirecell import units
from wirecell.util import ario

import numpy
import matplotlib.pyplot as plt

import json
import bz2


# the depo "data" arrays
#        0123456
columns="tqxyzLT"
# Note: some old format used another ordering and included "s" and
# "n".

# the depo "info" arrays have columns:
# (id, pdg, gen, child)


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
    keep = depos['s'] > 0.0
    for key in depos:
        depos[key] = depos[key][keep]
    return depos


def apply_units(depos, distance_unit, time_unit, energy_unit, step_unit=None, electrons_unit="1.0"):
    'Apply units to a deposition array, return a new one'

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


def stream(depofile, generation=0):
    '''
    Like load() but yield over indices.
    '''
    fp = ario.load(depofile)

    index = 0
    while True:
        try:
            yield load(fp, index, generation)
        except KeyError:
            return

        index += 1



def load(depofile, index=0, generation=0):
    '''Return depo info from file.

    The integer specifies the generation of depos with 0 being the
    "youngest" and its "prior" depo being 1 higher.  

    A dict holding seven items of array value is returned.  The keys
    are t=time, c=charge, x, y, z, L=sigma_L, T=sigma_T.  The .columns
    string of this module provides these keys as a string.

    If index or generation do not exist, KeyError is raised.
    '''
    if isinstance(depofile, str):
        fp = ario.load(depofile)
    else:
        fp = depofile
    dat = fp[f'depo_data_{index}']
    nfo = fp[f'depo_info_{index}']

    if dat.shape[0] == 7 and nfo.shape[0] == 4:
        dat = dat.T
        nfo = nfo.T

    indices = nfo[:,2] == generation
    thegen = dat[indices,:]
    if thegen.shape[0] == 0:
        raise KeyError(f'no generation {generation}')
    return todict(thegen)


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


def _abc_hist(a, b, c, da, db):
    '''
    Plot b vs a weighted by c in da x db bins.  Return axis
    '''
    a = a/da
    b = b/db

    amm = (numpy.min(a), numpy.max(a))
    bmm = (numpy.min(b), numpy.max(b))
    na = int(amm[1] - amm[0])
    nb = int(bmm[1] - bmm[0])
    print(f'bounds: {bmm}[{nb}] vs {amm}[{na}]')
    if na==0 or nb==0:
        print("no range on an axis")
        raise ValueError("no range on an axis")

    aedges = numpy.linspace(amm[0], amm[1], na)
    bedges = numpy.linspace(bmm[0], bmm[1], nb)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    hist, abins, bbins = numpy.histogram2d(a, b, bins=(aedges, bedges), weights=c)
    im = ax.imshow(numpy.ma.masked_where(hist==0, hist), interpolation='none',
                   extent=(amm[0], amm[1], bmm[1], bmm[0]))
    plt.colorbar(im, ax=ax)
    return ax

def plot_qxz(depos, output):
    'Plot colz q as X vs Z'
    q = numpy.abs(depos["q"])
    x = depos["x"]
    z = depos["z"]

    ax = _abc_hist(x, z, q, units.cm, units.cm)
    ax.set_title("depo electrons")
    ax.set_xlabel("X [cm]")
    ax.set_ylabel("Z [cm]")
    plt.savefig(output, dpi=300)

def plot_qxy(depos, output):
    'Plot colz q as X vs Y'
    q = numpy.abs(depos["q"])
    x = depos["x"]
    y = depos["y"]

    ax = _abc_hist(x, y, q, units.cm, units.cm)
    ax.set_title("depo electrons")
    ax.set_xlabel("X [cm]")
    ax.set_ylabel("Y [cm]")
    plt.savefig(output, dpi=300)

def plot_qzy(depos, output):
    'Plot colz q as Z vs Y'
    q = numpy.abs(depos["q"])
    z = depos["z"]
    y = depos["y"]

    ax = _abc_hist(z, y, q, units.cm, units.cm)
    ax.set_title("depo electrons")
    ax.set_xlabel("Z [cm]")
    ax.set_ylabel("Y [cm]")
    plt.savefig(output, dpi=300)

def plot_qxt(depos, output):
    'Plot colz q as X vs T'
    q = numpy.abs(depos["q"])
    x = depos["x"]
    t = depos["t"]

    ax = _abc_hist(x, t, q, units.cm, units.us)
    ax.set_title("depo electrons")
    ax.set_xlabel("X [cm]")
    ax.set_ylabel("T [us]")
    plt.savefig(output, dpi=300)

def plot_qzt(depos, output):
    'Plot colz q as Z vs T'
    q = numpy.abs(depos["q"])
    z = depos["z"]
    t = depos["t"]

    ax = _abc_hist(z, t, q, units.cm, units.us)
    ax.set_title("depo electrons")
    ax.set_xlabel("Z [cm]")
    ax.set_ylabel("T [us]")
    plt.savefig(output, dpi=300)

def plot_t(depos, output):
    'Plot t histogram weighted by q'
    q = numpy.abs(depos["q"])
    t = depos["t"]

    fig, ax = plt.subplots(1,1, tight_layout=True)
    ax.hist(t/units.us, bins=1000)
    ax.set_xlabel('T [us]')
    plt.savefig(output)

def plot_x(depos, output):
    'Plot x histogram weighted by q'
    q = numpy.abs(depos["q"])
    x = depos["x"]

    fig, ax = plt.subplots(1,1, tight_layout=True)
    ax.hist(x/units.cm, bins=1000)
    ax.set_xlabel('X [cm]')
    plt.savefig(output)

def _plot_abc(title, a, b, c, atit, btit, ctit, output, cmap='viridis'):
    '''
    Plot b vs a colored by c.
    '''
    cmap = plt.cm.get_cmap(cmap)

    # cmax = float(numpy.max(c))
    # colors = [cmap(cv/cmax) for cv in c]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)

    ax.set_xlabel(atit)
    ax.set_ylabel(btit)
    sc = ax.scatter(a, b, c=c, cmap=cmap)
    plt.colorbar(sc)
    fig.savefig(output)
    

def plot_xzqscat(depos, output):
    'Plot charge as scatter plot'

    _plot_abc('Charge',
              depos["x"]/units.mm, 
              depos["z"]/units.mm, 
              numpy.abs(depos["q"]),
              "x [mm]", "z [mm]", "q [ele]", output)

def plot_xyqscat(depos, output):
    'Plot charge as scatter plot'

    _plot_abc('Charge',
              depos["x"]/units.mm, 
              depos["y"]/units.mm, 
              numpy.abs(depos["q"]),
              "x [mm]", "y [mm]", "q [ele]", output)

def plot_tzqscat(depos, output):
    'Plot charge as scatter plot'

    _plot_abc('Charge',
              depos["t"]/units.us, 
              depos["z"]/units.mm, 
              numpy.abs(depos["q"]),
              "t [mus]", "z [mm]", "q [ele]", output)

def plot_tyqscat(depos, output):
    'Plot charge as scatter plot'

    _plot_abc('Charge',
              depos["t"]/units.us, 
              depos["y"]/units.mm, 
              numpy.abs(depos["q"]),
              "t [mus]", "y [mm]", "q [ele]", output)
