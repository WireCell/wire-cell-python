#!/usr/bin/env python
'''
Process Garfield field response output files to produce Wire Cell
field response input files.

Garfield input is provided as a tar file.  Internal directory
structure does not matter much but the files are assumed to be spelled
in the form:

    <impact>_<plane>.dat

where <impact> spells the impact position in mm and plane is from the
set {"U","V","Y"}.

Each .dat file may hold many records.  See parse_text_record() for
details of the assumptions made when parsing.

These quantities must be given explicitly:

    - speed :: the nominal drift speed

'''
from .. import units
from . import response

import numpy

import os.path as osp

from wirecell.util.fileio import load as source_loader

from wirecell.resp.garfield import dataset_asdict


def load(source, normalization = None, zero_wire_loc = 0.0, delay=0):
    '''Load Garfield data source (eg, tarball).

    Return list of response.ResponseFunction objects.

    The responses will be normalized according to the value of
    `normalization`:

    none or 0: no normalization, take Garfield as absolutely
    normalized

    less than 0: normalize such that the average of the integrals
    along an impact position of the central collection wire is this
    many electrons.

    greater than 0: simply multiply the responses by this number.

    The `zero_wire_loc` is the transverse location of the wire to be
    considered the central wire.

    '''
    source = source_loader(source, pattern="*.dat")

    uniq = dataset_asdict(source)

    # This following is a hack previously in the parser where it
    # definitely does not belong.  I move it here where it still
    # doesn't really belong but until we better factor this loader it
    # will have to do.
    if delay:
        # Delay the FR by given number of sample periods.
        print(f'delaying response by {delay}')
        for key in uniq:
            c = uniq[key]['y']
            uniq[key]['y'] = numpy.hstack((numpy.zeros(delay)*c[0], c[:-delay]))

    ret = list()
    for plane, zwl in zip('uvw', zero_wire_loc):
        byplane = [one for one in uniq.values() if one['plane'] == plane]
        zeros = [one for one in byplane if one['wire_region_pos'][0] == zwl and one['impact'] == 0.0]
        if len(zeros) != 1:
            for one in sorted(byplane, key=lambda x: (x['wire_region_pos'][0], x['impact'])):
                print ("region=%s impact=%s" % (one['wire_region_pos'], one['impact']))
            raise ValueError("%s-plane, failed to find exactly one zero wire (at %f).  Found: %s wires" % (plane.upper(), zwl, zeros))
        zero_wire_region = zeros[0]['wire_region']
        this_plane = list()
        for one in byplane:
            times = one['x']
            t0 = int(times[0])                         # fix round off
            tf = int(times[-1])                        # fix round off
            ls = (t0, tf, len(times))

            relative_region = one['wire_region'] - zero_wire_region
            fix_garfield_impact_sign = -1
            impact_number = fix_garfield_impact_sign * one['impact']
            rf = response.ResponseFunction(plane, relative_region,
                                           one['wire_region_pos'],
                                           ls, numpy.asarray(one['y']),
                                           impact_number)
            this_plane.append(rf)
        this_plane.sort(key=lambda x: x.region * 10000 + x.impact)
        ret += this_plane

    w0 = [r for r in ret if r.region == 0 and r.plane == 'w']
    dt = (w0[0].times[1]-w0[0].times[0])
    itot = sum([sum(w.response) for w in w0])/len(w0)
    qtot = itot*dt


    if normalization is None or normalization == 0:
        print ("No normalizing. But, %d paths (Qavg=%f fC = %f electrons)" % \
                   (len(w0), qtot/units.femtocoulomb, -qtot/units.eplus))

        return ret

    if normalization < 0:
        norm = normalization*units.eplus/qtot
        print ("Normalizing over %d paths is %f (dt=%.2f us, Qavg=%f fC = %f electrons)" % \
                   (len(w0), norm, dt/units.microsecond, qtot/units.femtocoulomb, -qtot/units.eplus))
    else:
        norm = normalization
        print ("Normalization by scaling with: %f" % norm)


    for r in ret:
        r.response *= norm

    return ret


def toarrays(rflist):
    '''
    Return field response current waveforms as 3 2D arrays.

    Return as tuple (u,v,w) where each is a 2D array shape:
    (#regions, #responses).
    '''
    ret = list()
    for byplane in response.group_by(rflist, 'plane'):
        this_plane = list()
        byregion = response.group_by(byplane, 'region')
        if len(byregion) != 1:
            raise ValueError("unexpected number of regions: %d" % len(byregion))
        for region in byregion:
            this_plane.append(region.response)
        ret.append(numpy.vstack(this_plane))
    return tuple(ret)



def convert(inputfile, outputfile = "wire-cell-garfield-fine-response.json.bz2", average=False, shaped=False):
    '''
    Convert an input Garfield file pack into an output wire cell field response file.

    See also wirecell.sigproc.response.persist
    '''
    rflist = load(inputfile)
    if shaped:
        rflist = [d.shaped() for d in rflist]
    if average:
        rflist = response.average(rflist)
    response.write(rflist, outputfile)

    

