'''
A module to produce paraview / vtk objects
'''

import math
import numpy
from collections import defaultdict
from wirecell import units

def undrift_points(pts, speed=1.6*units.mm/units.us, t0=0, time_index=0):
    '''
    Transform coordinate at time_index in pts from time to space.
    '''
    pts = numpy.array(pts)
    time = pts[:,time_index] + t0
    pts[:,time_index] = speed*time
    return pts


def undrift_depos(depos, speed=1.6*units.mm/units.us, time=0, drift_index=0):
    '''
    Remove the drift from the depos.

    The depos are as returned by wirecell.gen.depos.load().

    The time and drift coordinates will be changed to "back up" the
    depos to the given time and at the given speed. '''

    dt = depos['t'] - time
    depos['t'][:] = time
    depos['x'] = speed*dt
    depos['q'] = numpy.abs(depos['q'])
    return depos

def undrift_blobs(cgraph, speed=1.6*units.mm/units.us, time=0, x0=0, drift_index=0):
    '''Transform the blobs in the cluster graph.

    The cgraph may be a list of cluster graphs.

    The drift coordinates are changed from time to space.
    '''
    is_list=True
    if not isinstance(cgraph, list):
        cgraph = [cgraph]
        is_list = False

    ret = list()
    for gr in cgraph:
        for node, ndata in gr.nodes.data():
            if ndata['code'] != 'b':
                continue;

            pts = numpy.array(ndata['corners'])
            dt = pts[:,0] - time
            pts[:,0] = x0 - speed*dt;
            ndata['corners'] = pts
            ndata['span'] *= speed

        ret.append(gr)

    if is_list:
        return ret
    return ret[0]


def extrude(pts, dx):
    '''
    make a 3d set of cells based on ring of pts extruded along X axis by dx

    Return points and "relative cells"


    '''
    pts2 = [ [pt[0]+dx,pt[1],pt[2]] for pt in pts] # the other face
    all_pts = pts + pts2

    n = len(pts)
    top_cell = range(n)
    bot_cell = range(n, 2*n)
    cells = [top_cell, bot_cell]

    # enumerate the sides
    for ind in range(n):
        ind2 = (ind+1)%n
        cell = [top_cell[ind], top_cell[ind2], bot_cell[ind2], bot_cell[ind]]
        cells.append(cell)

    return all_pts, cells
        


def orderpoints(pointset):
    c = [0.0,0.0,0.0]
    for p in pointset:
        for i in range(3):
            c[i] += p[i]
    n = len(pointset)
    for i in range(3):
        c[i] /= n

    byang = list()
    for p in pointset:
        ang = math.atan2(p[2]-c[2], p[1]-c[1]);
        byang.append((ang, p))
    byang = sorted(byang, key=lambda x: x[0])
    return [p for a,p in byang]


def depos2pts(depos):
    '''
    Convert depos dict of numpy arrays tvtk unstructured grid.
    '''
    from tvtk.api import tvtk

    q = depos['q']
    npts = len(q)

    pts = numpy.vstack((depos['x'], depos['y'], depos['z'])).T

    indices = list(range(npts))

    ret = tvtk.PolyData(points=pts)
    verts = numpy.arange(0, npts, 1)
    verts.shape = (npts,1)
    ret.verts = verts
    ret.point_data.scalars = indices[:npts]
    ret.point_data.scalars.name = 'indices'

    ret.point_data.add_array(q)
    ret.point_data.get_array(1).name = 'charge'

    ret.point_data.add_array(depos['t'])
    ret.point_data.get_array(2).name = 'time'

    ret.point_data.add_array(depos['T'])
    ret.point_data.get_array(3).name = 'DT'

    ret.point_data.add_array(depos['L'])
    ret.point_data.get_array(4).name = 'DL'

    return ret



def clusters2blobs(gr):
    '''
    Given a graph object return a tvtk data object with blobs.
    '''
    from tvtk.api import tvtk

    all_points = list()
    blob_cells = list()
    datasetnames = set()
    values = list()
    for node, ndata in gr.nodes.data():
        if ndata['code'] != 'b':
            continue;
        vals = dict();
        thickness = 1.0
        for key,val in ndata.items():
            if key == 'corners':
                pts = orderpoints(val)
                continue
            if key == 'span':
                thickness = val
                continue
            if key == 'code':
                continue
            if key == 'bounds':
                # dimensionality too high to convert
                continue
            datasetnames.add(key)
            vals[key] = val;
        pts,cells = extrude(pts, thickness)
        all_points += pts
        blob_cells.append((len(pts), cells))
        values.append(vals)

    ugrid = tvtk.UnstructuredGrid(points = all_points);
    ptype = tvtk.Polyhedron().cell_type
    offset = 0
    for npts,cells in blob_cells:
        cell_ids = [len(cells)]
        for cell in cells:
            cell_ids.append(len(cell))
            cell_ids += [offset+cid for cid in cell]
        ugrid.insert_next_cell(ptype, cell_ids)
        offset += npts

    ugrid.cell_data.scalars = list(range(len(values)))
    ugrid.cell_data.scalars.name = "indices"
    
    narrays = 1
    for datasetname in sorted(datasetnames):
        arr = numpy.asarray([vals.get(datasetname, 0.0) for vals in values], dtype=float)
        ugrid.cell_data.add_array(arr)
        ugrid.cell_data.get_array(narrays).name = datasetname
        narrays += 1

    return ugrid

    

def get_blob(gr,node):
    for other in gr[node]:
        odat = gr.nodes[other]
        if odat['code'] == 'b':
            return other
    return None

def get_slice(gr, bnode):
    for other in gr[bnode]:
        odat = gr.nodes[other]
        if odat['code'] == 's':
            return other
    return None

def get_neighbors_oftype(gr, node, code, with_data=False):
    'Return neighbors of channel type'
    ret = list()
    for other in gr[node]:
        odat = gr.nodes[other]
        if odat['code'] == code:
            if with_data:
                ret.append((other,odat))
            else:
                retu.append(other)
    return ret

def clusters2views(gr):
    from tvtk.api import tvtk

    class Perwpid:
        def __init__(self):
            self.allind = list()
            self.allchs = list()
            self.values = list()
    perwpid = defaultdict(Perwpid)
    for node, ndata in gr.nodes.data():
        if ndata['code'] != 'm':
            continue;
        bnode = get_blob(gr, node)
        if bnode is None:
            continue

        #x = gr.nodes[bnode]['corners'][0][0] # "t"
        #eckses.append(x)
        snode = get_slice(gr, bnode)
        if snode is None:
            raise ValueError("bad graph structure")
        wpid = ndata['wpid']

        sdat = gr.nodes[snode]
        snum = sdat['ident']
        perwpid[wpid].allind.append(snum)

        chids = [d[1]["ident"] for d in get_neighbors_oftype(gr, node, 'c', True)]
        perwpid[wpid].allchs += chids

        ident2sigs = {str(s['ident']):s for s in sdat['signal']}
        val = sum([ident2sigs[str(chid)]['val'] for chid in chids])

        perwpid[wpid].values.append((snum,chids,val))

    all_imgdat=dict()
    for wpid, dat in perwpid.items():
        smin = min(dat.allind)
        smax = max(dat.allind)
        cmin = min(dat.allchs)
        cmax = max(dat.allchs)
        arr = numpy.zeros((smax-smin+1, cmax-cmin+1))
        for ind,chids,val in dat.values:
            for ch in chids:
                arr[ind - smin, ch - cmin] += val
        imgdat = tvtk.ImageData(spacing=(1,1,1), origin=(0,0,0))
        imgdat.point_data.scalars = arr.T.flatten()
        imgdat.point_data.scalars.name = 'activity'
        imgdat.dimensions = list(arr.shape)+[1]
        all_imgdat[wpid] = imgdat
    return all_imgdat

def blob_center(bdat):
    '''
    Return an array of one point at the center of the blob
    '''
    thickness = bdat['span']
    value = bdat['val']
    arr = numpy.asarray(bdat['corners'])

    npts = arr.shape[0]
    center = numpy.array([0.0]*4, dtype=float)
    center[:3] = numpy.sum(arr, 0) / npts
    center[0] += 0.5*thickness
    center[3] = value;
    return center
    

def blob_uniform_sample(bdat, density):
    '''
    Return an array of points uniformly sampled in the blob
    '''
    import random
    from shapely.geometry import Polygon, Point
    thickness = bdat['span']
    value = bdat['val']

    # z is x, y is y
    xstart = bdat['corners'][0][0]
    corners = [(cp[2],cp[1]) for cp in orderpoints(bdat['corners'])]
    npts = len(corners);
    pgon = Polygon(corners)
    nwant = max(1, int(pgon.area * thickness * density))
    pts = list()
    min_x, min_y, max_x, max_y = pgon.bounds

    while len(pts) != nwant:
        p = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
        if (p.within(pgon)):
            pts.append([random.uniform(xstart, xstart+thickness), p.y, p.x, value/nwant]);
    return numpy.asarray(pts)
            
        

def blobpoints(gr, sample_method=blob_center):
    '''
    return Nx4 array with rows made of x,y,z,q.
    '''
    arr = None

    for node, ndata in gr.nodes.data():
        if ndata['code'] != 'b':
            continue;
        one = sample_method(ndata)
        if arr is None:
            arr = one
        else:
            arr = numpy.vstack((arr, one))

    return arr
