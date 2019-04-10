'''
A module to produce paraview / vtk objects
'''

import math
import numpy
from collections import defaultdict
from tvtk.api import tvtk, write_data


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


def clusters2blobs(tap):
    '''
    Given an object build from a JSON file from JsonClusterTap (tap) return a tvtk data object.

    tap = json.load("clusters-0000.json");
    dat = imt.vtk.cluster2blobs(tap);
    tvtk.api.write_data(dat, "blobs-0000.vtu");
    '''

    all_points = list()
    blob_cells = list()
    datasetnames = set()
    values = list()
    for vtx in tap['vertices']:
        if vtx['type'] != 'b':
            continue;
        vals = dict();
        thickness = 1.0
        for key,val in vtx['data'].items():
            if key == 'corners':
                pts = orderpoints(val)
                continue
            if key == 'span':
                thickness = val
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
        arr = [vals.get(datasetname, 0.0) for vals in values]
        ugrid.cell_data.add_array(arr)
        ugrid.cell_data.get_array(narrays).name = datasetname
        narrays += 1

    return ugrid

    

    
