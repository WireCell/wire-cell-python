#!/usr/bin/env python

from wirecell import units

def unitify(val, unit=""):
    '''
    Convert val into WCT system of units.

    When val is a string it is evaluated in the context of the WCT
    unit definitions and if unit is given it is multiplied resulting
    in a number.

    When val is a list, tuple, or dict of unit values, the same
    structure is returned with strings evaluted.
    '''
    if val is None:
        return None
    if isinstance(val, list):
        return [unitify(one, unit) for one in val]
    if isinstance(val, tuple):
        return tuple([unitify(one, unit) for one in val])
    if isinstance(val, dict):
        return {k:unitify(v, unit) for k,v in val.items()}
    if not isinstance(val, str):
        raise TypeError(f'unsupported value type: {type(val)}')
    if unit:
        val += "*" + unit
    return eval(val, units.__dict__)

def unitify_parse(string):
    '''
    Parse a string with unit expressions and return its value.  String
    assumed to be a comman separated list of scalar values.  A list is
    returned.
    '''

    vals = [v.strip() for v in string.split(",") if v.strip()]

    vals = [eval(v, units.__dict__) for v in vals]
    return vals

