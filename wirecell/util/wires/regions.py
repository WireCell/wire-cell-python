#!/usr/bin/env python
'''
Functions to process descriptions of wire regions
'''

from wirecell import units
from collections import namedtuple

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
    

def uboone_shorted(filename):
    '''
    Load in the CSV file holding description of shorted wires in
    microboone.  Eg, one saved from Brooke's
    MicroBooNE_ShortedWireList.xlsx
    '''

    #### wirecell numbers:
    # Plane,Channel,Start X,Start Y,Start Z,End X,End Y,End Z,
    # Plane,Channel,Start X,Start Y,Start Z,End X,End Y,End Z,
    #### then larsoft numbers
    # Channel,Plane,Wire,Start X,Start Y,Start Z,End X,End Y,End Z,
    # Channel,Plane,Wire,Start X,Start Y,Start Z,End X,End Y,End Z


    regions = dict()
    current = None
    for lineno, line in enumerate(open(filename).readlines()):
        line=line.strip()

        if not line:
            continue
        vals = line.split(',')
        if not vals[0]:
            continue

        if "region" in vals[0].lower():
            rname = vals[0]
            print ("new region set: %s" % rname)
            current = regions[rname] = list()
            continue

        if vals[0].lower() == "plane":
            continue                      # column names

        wc = Region(wcborder(vals[0:8]), wcborder(vals[8:16]))
        ls = Region(lsborder(vals[16:25]), lsborder(vals[25:34]))
        current.append(WCLS(wc,ls))
    return regions

