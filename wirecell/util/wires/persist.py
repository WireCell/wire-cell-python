#!/usr/bin/env python
'''
Functions to assist in persisting schema objects.

'''

# fixme: except for this next line, this module is generic.  See also similar
# code for the detector response schema.
from . import schema

###########################

import json
import numpy


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
                return typ(**{k: fromdict(v) for k, v in obj[tname].items()})

    if isinstance(obj, list):
        return [fromdict(ele) for ele in obj]

    return obj


def dumps(obj, indent=2):
    '''
    Dump object to JSON text.
    '''
    return json.dumps(todict(obj), indent=indent)


def loads(text):
    '''
    Load object from JSON text.
    '''
    return fromdict(json.loads(text))


def dump(filename, obj, indent=2):
    '''
    Save a response object (typically response.schema.FieldResponse)
    to a file of the given name.
    '''
    btext = dumps(obj, indent=indent).encode()
    if filename.endswith(".json"):
        open(filename, 'wb').write(btext)
        return
    if filename.endswith(".json.bz2"):
        import bz2
        bz2.BZ2File(filename, 'w').write(btext)
        return
    if filename.endswith(".json.gz"):
        import gzip
        gzip.open(filename, "wb").write(btext)
        return
    raise ValueError("unknown file format: %s" % filename)


def load(filename):
    '''
    Return response.schema object representation of the data in the
    file of the given name.
    '''
    if filename.endswith(".json"):
        return loads(open(filename, 'r').read())

    if filename.endswith(".json.bz2"):
        import bz2
        return loads(bz2.BZ2File(filename, 'r').read())

    if filename.endswith(".json.gz"):
        import gzip
        return loads(gzip.open(filename, "rb").read())

    raise ValueError("unknown file format: %s" % filename)
