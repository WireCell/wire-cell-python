from wirecell import units
import matplotlib.pyplot as plt
import numpy

from collections import defaultdict

from matplotlib.patches import Rectangle, Ellipse
from matplotlib.collections import PatchCollection

def subplots(nrows=1, ncols=1, **kwds):
    return plt.subplots(nrows, ncols, tight_layout=True, **kwds)

def blob_nodes(cgraph, filter = lambda b : True):
    ret = list()
    for node, ndata in cgraph.nodes.data():
        if ndata['code'] == 'b' and filter(ndata):
            ret.append(ndata)
    return ret


def wires_pimpos(cgraph):
    '''Return array (N,2,3) with (N, ((3-origin), (3-pitch)))

    The 3-origin is some point along the WIP=0 wire direction and
    3-pitch is vector in pitch direction with magnitude that of the
    pitch.

    '''

    dat = dict()
    for node, ndata in cgraph.nodes.data():
        if ndata['code'] != 'w':
            continue
        ix = int(ndata['tailx']/units.mm)
        one = (ix, ndata['plane'], ndata['wip'],
               ndata['tailx'], ndata['taily'], ndata['tailz'],
               ndata['headx'], ndata['heady'], ndata['headz'])
        dat.append(one)
    dat = numpy.array(dat, dtype=float)
    # (N, 9)
    
    ret = list()
    while dat.size:
        ix = dat[0][0]
        one = dat[ dat[:0] == ix, :]
        dat = dat[ dat[:0] != ix, :]

        wmin = one[numpy.argmin(one[:,2]),:][0]
        wmax = one[numpy.argmax(one[:,2]),:][0]
        dwip = wmax[2] - wmin[2]
        # wire centers
        c1 = 0.5*(wmin[3:6] + wmin[6:9])
        c2 = 0.5*(wmax[3:6] + wmax[6:9])

        # wire as ray
        r = wmin[6:9] - wmin[3:6]
        ecks = numpy.array([1,0,0])
        # pitch
        p = numpy.cross(ecks, r)
        p = p/numpy.linalg.norm(p)
        p *= numpy.dot(p, c2-c1)/dwip
        c = c1 - p*wmin[2]
        ret.append((c,p))
    return numpy.array(ret)

def blob_faces(blobs):
    return numpy.array([b['faceid'] for b in blobs], dtype=int)

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

def blob_corners(blobs):
    ''' Return (3,2,Nblobs) array of (min,max) bounds of a blob in
    each of the 3 Cartesian coordinates.  Corners X in spatial.
    '''
    spans = set()

    mm = numpy.zeros((3,2, len(blobs)))
    for ind, b in enumerate(blobs):
        c = numpy.array(b['corners'])
        span = b['span']
        spans.add(span)
        for axis in [0,1,2]:
            cmin = numpy.min(c[:,axis])
            mm[axis][0][ind] = cmin
            if axis:
                mm[axis][1][ind] = numpy.max(c[:,axis]) - cmin
            else:
                mm[axis][1][ind] = span

    spans = [s/units.cm for s in spans]
    print(f'spans: {spans} cm')
    return mm
    
def blob_bounds(blobs):
    '''Return (3,2,Nblobs) array of (min,max) WIP bounds for each view
    '''
    mm = numpy.zeros((len(blobs), 3, 2), dtype=int)
    for ind, b in enumerate(blobs):
        bounds = numpy.array(b['bounds'])
        mm[ind] = bounds
    mm = numpy.transpose(mm, [1,2,0])
    return mm
    

def blob_slices(blobs):
    '''Return (2,Nblobs) array of pair of slices' time (start, span)'''
    ss = numpy.zeros((2, len(blobs)))
    for ind, b in enumerate(blobs):
        ss[0][ind] = b['start']
        ss[1][ind] = b['span']
    return ss

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

    alpha = 0.5
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
    # (axis=3,start_or_span=2,N)
    bmm = blob_corners(blobs)/units.cm 
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

        # blob bounds
        bxs = bmm[ix,0,:]
        bys = bmm[iy,0,:]
        bws = bmm[ix,1,:]
        bhs = bmm[iy,1,:]

        # limits
        if lims is None:
            lim = numpy.zeros((2,2))

            # (axis, minmax)
            lim = [(numpy.min(bxs-bws), numpy.max(bxs+2*bws)),
                   (numpy.min(bys-bhs), numpy.max(bys+2*bhs))]
        else:
            lim = [lims[ix],
                   lims[iy]]
        lim = numpy.array(lim)
        print(f'lim={lim}')

        def inbounds(x,y):
            # return [True] * x.size
            return numpy.logical_and(
                numpy.logical_and(x >= lim[0,0], x <= lim[0,1]),
                numpy.logical_and(y >= lim[1,0], y <= lim[1,1]))

        # blobs
        bpc = None
        if "blobs" in include:
            bws = bmm[ix,1,:]
            bhs = bmm[iy,1,:]
            bps = [Rectangle((x,y),w,h) for x,y,w,h in zip(bxs, bys, bws, bhs)]
            bpc = PatchCollection(bps, alpha=alpha, cmap='viridis')
            bpc.set_array(bq)
            ax.add_collection(bpc)
            bqtot = numpy.sum(bq)
            print(f'total blob charge: {bqtot*1e-6:.1f}M')

        dpc = None
        if "depos" in include:
            didx = inbounds(dxys[:,0], dxys[:,1])
            dws = 2*depo_dr[ix,didx]*nsigma
            dhs = 2*depo_dr[iy,didx]*nsigma
            dxys = dxys[didx]
            dq =  depos['q'][didx]
            # many depos span range making elipses invisible
            if dq.size < 100:
                dps = [Ellipse(xy,w,h) for xy,w,h in zip(dxys, dws, dhs)]
                dpc = PatchCollection(dps, alpha=alpha, cmap='viridis')
                dpc.set_array(dq)
                ax.add_collection(dpc)
            dqtot = numpy.sum(dq)
            print(f'total depo charge: {dqtot*1e-6:.3f}M (in range)')
            ax.scatter(dxys[:,0], dxys[:,1], s=0.01, c='gray', marker=',', alpha=1)

        ax.set_xlim(lim[0])
        ax.set_ylim(lim[1])
        ax.set_xlabel(f'{l1} cm')
        ax.set_ylabel(f'{l2} cm')
        fig.colorbar(bpc or dpc, ax=ax)
    return fig



def plot_views(depos, cgraph):
    '''
    Plot blobs and depos by views (and slices)
    '''
    ## depos
    sigma_map = dict(x='L', y='T', z='T', u='T', v='T', w='T')
    depo_r = numpy.vstack((depos['x'], depos['y'], depos['z']))/units.cm
    depo_dr = numpy.vstack((depos['L'], depos['T'], depos['T']))/units.cm

    # cps = wire_pimpos(cgraph)
    

    ## blobs
    blobs = blob_nodes(cgraph)#, lambda b: b['value'] >= 0)
    bbs = blob_bounds(blobs) # (3,2,N), WIP bounds
    bq = blob_charge(blobs)  # (N,)
    bss = blob_slices(blobs) # (2,N), time (start,span)
    bfs = blob_faces(blobs)  # (N,)

    tstart = bss[0]
    tspan = bss[1]
    tmin = numpy.min(tstart)
    tmax = numpy.max(tstart+tspan)

    ntbins = int((tmax-tmin)/tspan[0])
    tbins = numpy.array((tstart-tmin)/tspan[0], dtype=int)

    # pitch vs slice
    fig, axes = subplots(4,2, sharex=True)

    print (bbs.shape, bss.shape)

    for iview in [0,1,2]:
        letter = "UVW"[iview]

        for iface in [0,1]:
            ax = axes[iview,iface]
            
            find = bfs[bfs==iface]
            qs = bq[find]
            ts = tbins[find]
            wbins = bbs[iview,:,find]

            wipmin = wbins[0]
            wipmax = wbins[1]
            wmin = numpy.min(wipmin)
            wmax = numpy.max(wipmax)
            nwbins = int(wmax-wmin)

            arr = numpy.zeros((nwbins, ntbins))
            print(wbins.shape, ts.shape, qs.shape)
            for (w1,w2),t,q in zip(wbins, ts, qs):
                w1 -= wmin
                w2 -= wmin
                #print('w=',(w1,w2),'t=',(t1,t2),'q=',q)
                arr[w1:w2, t:t+1] += q

            im = ax.imshow(arr,
                           extent=(tmin,tmax, wmax, wmin),
                           aspect='auto', interpolation='none', cmap='viridis')
            ax.set_xlabel(f'T [us]')
            ax.set_ylabel(f'{letter} [WIP]')
            if iview == 0:
                ax.set_title(f'face {iface}')
            fig.colorbar(im, ax=ax)

    for iface in [0,1]:
        ax = axes[3,iface]
        find = bfs[bfs==iface]
        qs = bq[find]
        ts = tbins[find]

        print (f'ntbins={ntbins}, trange={(tmin,tmax)}, ts={ts}')
        ax.hist(ts+tmin, bins=ntbins, range=(tmin,tmax), weights=qs)

    return fig
