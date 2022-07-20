from wirecell import units
import matplotlib.pyplot as plt
import numpy

from matplotlib.patches import Rectangle, Ellipse
from matplotlib.collections import PatchCollection

def subplots(nrows=1, ncols=1):
    return plt.subplots(nrows, ncols, tight_layout=True)

def blob_nodes(cgraph):
    ret = list()
    for node, ndata in cgraph.nodes.data():
        if ndata['code'] == 'b':
            ret.append(ndata)
    return ret


def blob_coord(blobs, axis=0):
    ret = list()
    for b in blobs:
        ret.append(b['corners'][0][axis])
    return numpy.array(ret)
     
def blob_charge(blobs):
    ret = numpy.zeros(len(blobs))
    for ind, b in enumerate(blobs):
        ret[ind] = b['value']
    return ret
    

def blob_centers(blobs, index=2):
    ret = list()
    for b in blobs:
        c = 0
        corners = b['corners']
        for corner in corners:
            c += corner[index]
        cen = c/len(corners)
        ret.append(cen)
    return numpy.array(ret)

def blob_bounds(blobs):
    ''' Return (3,2,Nblobs) array of (min,max) bounds of a blob in
    each of the 3 Cartesian coordinates
    '''
    mm = numpy.zeros((3,2, len(blobs)))
    for ind, b in enumerate(blobs):
        c = numpy.array(b['corners'])
        for axis in [0,1,2]:
            mm[axis][0][ind] = numpy.min(c[:,axis])
            if axis:
                mm[axis][1][ind] = numpy.max(c[:,axis]) - mm[axis][0][ind]
            else:
                mm[axis][1][ind] = b['span']
    return mm
    

def plot_xz(depos, cgraph):
    fig, ax = subplots(1,1)
    blobs = blob_nodes(cgraph)
    ax.scatter(blob_coord(blobs)/units.cm, blob_centers(blobs)/units.cm,
               marker='s', label="blobs")
    ax.scatter(depos['x']/units.cm, depos['z']/units.cm,
               marker='.', label="depos")

    ax.set_title("depos and blobs")
    ax.set_xlabel("X [cm]")
    ax.set_ylabel("Z [cm]")    
    ax.legend()
    return fig

def plot_outlines(depos, cgraph, lims=None, include=("depos","blobs")):
    '''Plot depos as ellipses and blobs as squares.
    '''
    if include == "both" or "both" in include:
        include=("depos","blobs")
    print("including:",include)

    alpha = 1.0
    cscale = 10.0
    nsigma = 3.0

    ## depos
    sigma_map = dict(x='L', y='T', z='T', u='T', v='T', w='T')
    depo_r = numpy.vstack((depos['x'], depos['y'], depos['z']))/units.cm
    depo_dr = numpy.vstack((depos['L'], depos['T'], depos['T']))/units.cm

    # (3,N)
    print ('depos_r:', depo_r.shape, numpy.min(depo_r, axis=1), numpy.max(depo_r, axis=1))

    ## blobs
    blobs = blob_nodes(cgraph)
    bmm = blob_bounds(blobs)/units.cm # (3,2,N)
    #print('bmm:', bmm.shape)
    bq = blob_charge(blobs)

    fig, axes = subplots(1,3)

    for ix, iy in zip([0,1,2], [1,2,0]):
        ax = axes[ix]
        l1 = "XYZ"[ix]
        l2 = "XYZ"[iy]

        # depos
        dxys = None
        if "depos" in include:
            dxys = numpy.vstack((depo_r[ix], depo_r[iy])).T

        bxys = None
        if "blobs" in include:
            bxys = numpy.vstack((bmm[ix,0,:], bmm[iy,0,:])).T

        # limits
        if lims is None:
            lim = numpy.zeros((2,2))
            lxys = bxys
            if lxys is None:
                lxys = dxys
            lim = [(numpy.min(lxys[:,0]), numpy.max(lxys[:,0])),
                   (numpy.min(lxys[:,1]), numpy.max(lxys[:,1]))]
        else:
            lim = [lims[ix], lims[iy]]
        lim = numpy.array(lim)

        def inbounds(x,y):
            return numpy.logical_and(
                numpy.logical_and(x >= lim[0,0], x <= lim[0,1]),
                numpy.logical_and(y >= lim[1,0], y <= lim[1,1]))

        # blobs
        bpc = None
        if "blobs" in include:
            bws = bmm[ix,1,:]
            bhs = bmm[iy,1,:]
            bps = [Rectangle(xy,w,h) for xy,w,h in zip(bxys, bws, bhs)]
            bpc = PatchCollection(bps, alpha=alpha, cmap='viridis')
            bpc.set_array(bq)
            ax.add_collection(bpc)
            bqtot = numpy.sum(bq)
            print(f'total blob charge: {bqtot*1e-6:.1f}M')

        dpc = None
        if "depos" in include:
            didx = inbounds(dxys[:,0], dxys[:,1])
            dxys = dxys[didx]
            dws = depo_dr[ix,didx]*nsigma
            dhs = depo_dr[iy,didx]*nsigma
            dq = -depos['q'][didx]
            dps = [Ellipse(xy,w,h) for xy,w,h in zip(dxys, dws, dhs)]
            dpc = PatchCollection(dps, alpha=alpha, cmap='viridis')
            dpc.set_array(dq)
            ax.add_collection(dpc)
            dqtot = numpy.sum(dq)
            print(f'total depo charge: {dqtot*1e-6:.3f}M (in range)')

        ax.set_xlim(lim[0])
        ax.set_ylim(lim[1])
        ax.set_xlabel(l1)
        ax.set_ylabel(l2)
        fig.colorbar(bpc or dpc, ax=ax)
