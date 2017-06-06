#!/usr/bin/env python

from wirecell import units

def unitify(valstr, unit=""):
    if unit:
        valstr += "*" + unit
    return eval(valstr, units.__dict__)
