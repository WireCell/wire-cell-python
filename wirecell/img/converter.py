'''
A module to produce paraview / vtk objects
'''

import math
import numpy
from collections import defaultdict


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
    byang.sort()
    return [p for a,p in byang]


def depos2pts(arr):
    '''
    Convert numpy array like which comes from 'depo_data_0' key of npz
    file from NumpyDepoSaver to tvtk unstructured grid.
    '''
    from tvtk.api import tvtk, write_data

    npts = arr.shape[1]
    # t,q,x,y,z,dl,dt
    q = arr[1,:].reshape(npts)
    pts = arr[2:5,:].T

    indices = list(range(npts))

    ret = tvtk.PolyData(points=pts)
    verts = numpy.arange(0, npts, 1)
    verts.shape = (npts,1)
    ret.verts = verts
    ret.point_data.scalars = indices[:npts]
    ret.point_data.scalars.name = 'indices'

    ret.point_data.add_array(q)
    ret.point_data.get_array(1).name = 'charge'

    return ret



def clusters2blobs(gr):
    '''
    Given a graph object return a tvtk data object with blbos.
    '''
    from tvtk.api import tvtk, write_data

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


def clusters2views(gr):
    from tvtk.api import tvtk, write_data

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
            raise ValueError("bad graph structure")
        #x = gr.nodes[bnode]['corners'][0][0] # "t"
        #eckses.append(x)
        snode = get_slice(gr, bnode)
        if snode is None:
            raise ValueError("bad graph structure")
        wpid = ndata['wpid']
        

        sdat = gr.nodes[snode]
        snum = sdat['ident']
        perwpid[wpid].allind.append(snum)
        sact = sdat['activity']
        val = 0.0
        chids = ndata['chids']
        perwpid[wpid].allchs += chids
        for chid in chids:
            val += float(sact[str(chid)])
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
    value = bdat['value']
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
    value = bdat['value']

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
