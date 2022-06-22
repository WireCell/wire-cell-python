#!/usr/bin/env python

from wirecell import units

def unitify(valstr, unit=""):
    '''
    Return a numeric value from a string holding a unit expression or by providing a unit.
    '''

    if unit:
        valstr += "*" + unit
    return eval(valstr, units.__dict__)

def unitify_parse(string):
    '''
    Parse a string with unit expressions and return its value.  String
    assumed to be a comman separated list of scalar values.  A list is
    returned.
    '''

    vals = [v.strip() for v in string.split(",") if v.strip()]

    vals = [eval(v, units.__dict__) for v in vals]
    return vals

