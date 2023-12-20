#!/usr/bin/env python
'''
Functions to assist in persisting schema objects.
'''
from . import schema
from wirecell.util import detectors

###########################

import json
import numpy

from wirecell.util import jsio


def todict(obj):
    '''
    Return a dictionary for the object which is marked up for type.
    '''
    for typename in [c.__name__ for c in schema.classes()]:
        if typename == type(obj).__name__:
            cname = obj.__class__.__name__
            return {cname: {k: todict(v) for k, v in obj._asdict().items()}}
    if isinstance(obj, numpy.ndarray):
        shape = list(obj.shape)
        elements = obj.flatten().tolist()
        return dict(array=dict(shape=shape, elements=elements))
    if isinstance(obj, list):
        return [todict(ele) for ele in obj]

    return obj


def fromdict(obj):
    '''
    Undo `todict()`.
    '''
    if isinstance(obj, dict):

        for typ in schema.classes():
            tname = typ.__name__
            if tname in obj:

                # The "detectors" attribute was added to the schema
                # and some older files may not include it.
                if tname == "Store" and "detectors" not in obj["Store"]:
                    obj["Store"]["detectors"] = [schema.Detector(
                        ident=0,
                        anodes=list(range(len(obj["Store"]["anodes"]))))]

                return typ(**{k: fromdict(v) for k, v in obj[tname].items()})

    if isinstance(obj, list):
        return [fromdict(ele) for ele in obj]

    return obj


dumps = jsio.dumps
loads = jsio.loads
dump = jsio.dump

def load(name):
    '''Return wires schema object representation.

    The name may be that of a "wires file" or it may provide a canonical
    detector name (eg "pdsp", "uboone") in which case the detectors registry
    will be used to resolve that to a wires file name to load.

    '''

    if '.json' in name:
        return fromdict(jsio.load(name))

    return fromdict(detectors.load(name, "wires"))

