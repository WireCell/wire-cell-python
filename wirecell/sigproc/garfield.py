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
import tarfile

import os.path as osp

# fixme: move to some util module
def fromtarfile(filename):
    '''
    Iterate on tarfile, returning (name,text) pair of each file.
    '''
    tf = tarfile.open(filename, 'r')
    for name,member in sorted([(m.name,m) for m in tf.getmembers()]):
        if member.isdir():
            continue
        yield (member.name, tf.extractfile(member).read().decode('utf-8'))
# fixme: move to some util module
def asgenerator(source):
    '''
    If string, assume file, open proper generator, o.w. just return
    '''
    if type(source) not in [type(""), type(u"")]:
        #print ('Source is not a string: %s' % type(source))
        return source
    if osp.splitext(source)[1] in [".tar", ".gz", ".tgz"]:
        return fromtarfile(source)
    raise ValueError('unknown garfield data source: "%s"' % source)


def split_text_records(text):
    '''
    Return a generator that splits text by record separators.
    '''
    for maybe in text.split("\n% "):
        if maybe.startswith("Created"):
            yield maybe

def parse_text_record(text):
    '''
    Iterate on garfield text, returning one record.
    '''
    lines = text.split('\n')

    ret = dict()

    # Created 31/07/16 At 19.52.20 < none > SIGNAL   "Direct signal, group   1     "
    created = lines[0].split()
    ret['created'] = '%s %s' %(created[1], created[3])
    ret['signal'] = None
    if 'Direct signal' in lines[0]:
        ret['signal'] = 'direct'
    if 'Cross-talk' in lines[0]:
        ret['signal'] = 'x-talk'

    #   Group 1 consists of:
    ret['group'] = int(lines[2].split()[1])

    #      Wire 243 with label X at (x,y)=(-3,0.6) and at -110 V
    wire = lines[3].split()
    ret['wire_region'] = int(wire[1])
    ret['label'] = wire[4]

    pos = map(float, wire[6].split('=')[1][1:-1].split(','))
    ret['wire_region_pos'] = tuple([p*units.cm for p in pos])
    ret['bias_voltage'] = float(wire[9])

    #  Number of signal records:  1000
    ret['nbins'] = nbins = int(lines[4].split()[4])

    #  Units used: time in micro second, current in micro Ampere.
    xunit, yunit = lines[5].split(":")[1].split(",")
    xunit = [x.strip() for x in xunit.split("in")]
    yunit = [y.strip() for y in yunit.split("in")]

    xscale = 1.0 # float(lines[7].split("=")[1]);
    if "micro second" in xunit[1]:
        xscale = units.us

    yscale = 1.0 # float(lines[8].split("=")[1]);
    if "micro Ampere" in yunit[1]:
        yscale = units.microampere

    ret['xlabel'] = xunit[0]
    ret['ylabel'] = yunit[0]

    xdata = list()
    ydata = list()
    #  + (  0.00000000E+00   0.00000000E+00
    #  +     0.10000000E+00   0.00000000E+00
    # ...
    #  +     0.99800003E+02   0.00000000E+00
    #  +     0.99900002E+02   0.00000000E+00 )
    for line in lines[9:9+nbins]:
        xy = line[4:].split()
        xdata.append(float(xy[0]))
        ydata.append(float(xy[1]))
    if nbins != len(xdata) or nbins != len(ydata):
        raise ValueError('parse error for "%s"' % wire)
    ret['x'] = numpy.asarray(xdata)*xscale
    ret['y'] = numpy.asarray(ydata)*yscale
    return ret


def parse_filename(filename):
    '''
    Try to parse whatever data is encoded into the file name.
    '''
    fname = osp.split(filename)[-1]
    dist, plane = osp.splitext(fname)[0].split('_')
    plane = plane.lower()
    if plane == 'y':
        plane = 'w'
    return dict(impact=float(dist), plane=plane, filename=filename)

def load(source, normalization = None, zero_wire_loc = 0.0):
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
    source = asgenerator(source)

    from collections import defaultdict
    uniq = defaultdict(dict)

    for filename, text in source:

        fnamedat = parse_filename(filename)

        gen = split_text_records(text)
        for rec in gen:
            dat = parse_text_record(rec)

            key = tuple([filename] + [dat[k] for k in ['group', 'wire_region', 'label']])

            old = uniq.get(key, None)
            if old:             # sum up both signal types
                ### debugging: central collection wire has both records but one is miniscule
                #oldtot = sum(old['y'])
                #if abs(oldtot) > 0.0: 
                #    print ('found key already: "%s", sum=%e+%e"' %(key, sum(dat['y']), sum(old['y'])))
                old['y'] += dat['y']
                continue

            dat.pop('signal')
            dat.update(fnamedat)
            uniq[key] = dat

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

    

