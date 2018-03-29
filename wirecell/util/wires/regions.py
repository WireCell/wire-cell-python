#!/usr/bin/env python
'''
Functions to process descriptions of wire regions
'''

from wirecell import units
from collections import namedtuple, defaultdict

Point = namedtuple("Point","x y z")
Border = namedtuple("Border","plane wire ch tail head")   # a wire
Region = namedtuple("Region","beg end")   # pair of borders
WCLS = namedtuple("WCLS", "wc ls")

def wcborder(vals):
    '''
    convert array of
    Plane,Channel,Start X,Start Y,Start Z,End X,End Y,End Z,
    to Border object
    '''
    nums = [int(v) for v in vals[:2]]
    coords = [round(1000*float(v)*units.cm)/1000.0 for v in vals[2:]]

    return Border(nums[0], nums[1], nums[1],
                      Point(*coords[0:3]), Point(*coords[3:6]))
def lsborder(vals):
    '''
    convert array of
    # Channel,Plane,Wire,Start X,Start Y,Start Z,End X,End Y,End Z,
    to Border object
    '''
    nums = [int(v) for v in vals[:3]]
    coords = [round(1000*float(v)*units.cm)/1000.0 for v in vals[3:]]

    return Border(nums[1], nums[2], nums[0],
                      Point(*coords[0:3]), Point(*coords[3:6]))
    

def uboone_shorted(store, filename):
    '''
    Load in the CSV file holding description of shorted wires in
    microboone.  Confirm data is consistent with given
    wires.schema.Store object.

    Example file is the CSV saved from MicroBooNE_ShortedWireList.xlsx.

    Return data structure describing the shorted wires.
    '''

    #### wirecell numbers:
    # Plane,Channel,Start X,Start Y,Start Z,End X,End Y,End Z,
    # Plane,Channel,Start X,Start Y,Start Z,End X,End Y,End Z,
    #### then larsoft numbers
    # Channel,Plane,Wire,Start X,Start Y,Start Z,End X,End Y,End Z,
    # Channel,Plane,Wire,Start X,Start Y,Start Z,End X,End Y,End Z

    
    # return a dictionary indexed by shorted plane number and with
    # values which are lists of triples of (plane,wire1,wire2)

    ret = defaultdict(list)
    last_triple = list()
    shorted_plane = None
    for lineno, line in enumerate(open(filename).readlines()):
        line=line.strip()

        if not line:
            continue
        vals = line.split(',')
        if not vals[0]:
            continue

        maybe = vals[0].lower();
        if "shorted region" in maybe:
            letter = maybe[maybe.find("shorted region")-2]
            #print 'Starting plane "%s"' % letter
            shorted_plane = "uvy".index(letter)
            continue

        if vals[0].lower() == "plane":
            continue                      # column names

        plane = int(vals[17])

        ch1 = int(vals[16])
        w1 = int(vals[18])
        ch2 = int(vals[25])
        w2 = int(vals[27])

        chw1 = [w.ident for w in store.wires if w.channel == ch1]
        chw2 = [w.ident for w in store.wires if w.channel == ch2]
        assert w1 in chw1, "w1 %s %s"%(w1,chw1)
        assert w2 in chw2, "w2 %s %s"%(w2,chw2)
        one = dict(plane=plane,ch1=ch1,wire1=w1,ch2=ch2,wire2=w2)
        if plane == shorted_plane:
            print "[ {plane:%d, min:%d, max:%d} ]," % (plane,w1,w2)
        last_triple.append(one)

        if plane == 2:
            if last_triple:
                ret[shorted_plane].append(last_triple)
            last_triple = list()

    return ret

        


