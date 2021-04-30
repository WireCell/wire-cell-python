#!/usr/bin/env python3
'''
Support for garfield files
'''

import numpy
import os.path as osp
from collections import defaultdict
from wirecell import units

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

    Faithfully return data AS IS with no modification other than to
    convert to WCT system of units.
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


def dataset_asdict(source):
    '''
    Return a dictionary representation of a garfield dataset source

    Source is a sequence of (file name, file text)

    Returned dict is keyed by tuple of Garfield meta info:

    (filename, group, region, label)

    This tuple is used as a key to assure uniqueness.  The value for a
    key also includes entries for the last three ntuple elements.
    '''

    uniq = defaultdict(dict)

    for filename, text in source:

        try:
            fnamedat = parse_filename(filename)
        except ValueError as ve:
            print(f'fail to parse {ve}, skip {filename}')
            continue

        gen = split_text_records(text)
        for rec in gen:
            dat = parse_text_record(rec)

            key = tuple([filename] + [dat[k] for k in ['group', 'wire_region', 'label']])

            old = uniq.get(key, None)
            if old:             # sum up both signal types
                old['y'] += dat['y']
                continue

            dat.pop('signal')
            dat.update(fnamedat)
            uniq[key] = dat

    return uniq


def dsdict_dump(ds):
    '''
    Print info about a garfield dataset dict
    '''
    planes = defaultdict(list)
    for d in ds.values():
        planes[d['plane']].append(d)

    for plane, dats in planes.items():
        wires = [d['wire_region'] for d in dats]
        imps = [d['impact'] for d in dats]
        print (f'{plane}: wires: {len(set(wires))} (tot:{len(wires)}), imps: {len(set(imps))} (tot:{len(imps)})')
        print ('\titem 0:')
        for k,v in dats[0].items():
            if isinstance(v, numpy.ndarray):
                print(f'\t{k}: {v.shape}')
            else:
                print(f'\t{k}: {v}')



def dsdict2arrays(ds, speed, origin):
    '''
    Return official array representation of a garfield dataset source

    Garfield does not tell us speed nor origin so you MUST get it right.
    '''

    byplane = defaultdict(list)
    for d in ds.values():
        byplane[d['plane']].append(d)

    arrays = dict()
    for plane, values in byplane.items():
        nwires = len(set([d['wire_region'] for d in values]))
        nimps = len(set([d['impact'] for d in values]))

        data = list()
        for val in values:
            fix_garfield_impact_sign = -1
            impact = fix_garfield_impact_sign * val['impact']
            pitchpos = val['wire_region_pos'][0]
            current = val['y']
            data.append(((pitchpos, impact), current))
        data.sort()
        current = numpy.vstack([d[-1] for d in data])

        # pitch location of wire and relative impact position (N,2)
        rpi = numpy.asarray([d[0] for d in data])
        pitchpos = rpi[:,0] + rpi[:,1]
        
        pitches = list(set([d[0][0] for d in data]))
        pitches.sort()
        pitch = pitches[1] - pitches[0]
        location = val['wire_region_pos'][1]
        ticks = val['x']
        full_time = ticks[-1] - ticks[0]
        period = full_time / (ticks.size - 1)

        # flip to put positive pitchpos at top of "matrix"
        arrays[f'current_{plane}'] = numpy.flip(current, axis=0)
        arrays[f'pitchpos_{plane}'] = numpy.flip(pitchpos)
        arrays[f'plane_{plane}'] = numpy.array([origin, location, pitch, speed, period])
        # we can't supply wirepos nor paths

    return arrays
