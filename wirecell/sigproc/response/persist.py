#!/usr/bin/env python
'''
Functions to handle serializing of field response data.

There are four transient respresentations of FR:

- schema :: named tuple objects as described by .schema similar to that of WCT C++.

- pod :: dict of POD with structure similar to .schema

- arrays :: dict of arrays similar to pod but with numpy arrays instead of lists and as described in .arrays

- list :: a list of elements of one of the above types.

The "schema" representation may be converted to/from the other two. 

There are three persistent representations

- json :: essentially json.dumps() called on the "pod" (or "list of pod") representation.

- numpy :: essentially numpy.savez() called on the "arrays" representation.  This does not support "list".

- name :: the default field responses may be loaded (but not dumped) by a canonical detector name.

The persistent representations may be compressed.

'''

from pathlib import Path
import numpy

from .schema import FieldResponse, PlaneResponse, PathResponse
from . import arrays as frarrs
from wirecell.util import detectors, jsio

def is_schema(obj):
    '''
    Return true if obj is in schema rep.
    '''
    return isinstance(obj, FieldResponse)


def is_pod(obj):
    '''
    Return true if obj is in dict of pod rep.
    '''
    return isinstance(obj, dict) and 'FieldResponse' in obj


def is_array(obj):
    '''
    Return true if obj is in array rep.
    '''
    return isinstance(obj, dict) and 'origin' in obj


def is_list(obj):
    '''
    Return true if obj is in list rep.
    '''
    return isinstance(obj, list)

def schema2pod(obj):
    '''
    Convert FR from schema to pod representations or lists thereof.
    '''
    for typename in ['FieldResponse', 'PlaneResponse', 'PathResponse']:
        if typename == type(obj).__name__:
            cname = obj.__class__.__name__
            return {cname: {k: schema2pod(v) for k, v in obj._asdict().items()}}

    if isinstance(obj, numpy.ndarray):
        shape = list(obj.shape)
        elements = obj.flatten().tolist()
        return dict(array=dict(shape=shape, elements=elements))

    if is_list(obj):
        return [schema2pod(ele) for ele in obj]

    return obj


def pod2schema(obj):
    '''
    Convert FR from pod to schema representations or lists thereof.

    This function is actually recursive on the pod structure.
    '''
    if is_list(obj):
        return [pod2schema(ele) for ele in obj]

    if isinstance(obj, dict):

        if 'array' in obj:
            ret = numpy.asarray(obj['array']['elements'])
            return ret.reshape(obj['array']['shape'])

        for typ in [FieldResponse, PlaneResponse, PathResponse]:
            tname = typ.__name__
            if tname in obj:
                tobj = obj[tname]
                return typ(**{k: pod2schema(v) for k, v in tobj.items() if k not in ["pitchdir","wiredir"]})

    return obj


def dump_fr(obj, msg=""):
    if isinstance(obj, dict):
        print(f'DUMP FR: {type(obj)} {obj.keys()} {msg}')
    else:
        print(f'DUMP FR: {type(obj)} {msg}')


def schema2array(obj):
    '''
    Convert FR from schema to array representations, or lists thereof.
    '''
    dump_fr(obj, "schema2array")

    if is_schema(obj):
        return frarrs.toarray(obj)

    if is_list(obj):
        return [schema2array(one) for one in obj]

    raise TypeError(f'expecting schema representation for FR, given {type(obj)}')


def array2schema(obj):
    if is_array(obj):
        got = frarrs.toschema(obj)
        print(f'array2schema: {type(obj)} -> {type(got)}')
        return got
    if is_list(obj):
        return [frarrs.toschema(one) for one in obj]
    raise TypeError(f'expecting array representation for FR, given {type(obj)}')



def topod(obj):
    '''
    Return a pod representation of an FR object.
    '''
    dump_fr(obj, "topod")

    if is_pod(obj):
        return obj

    if is_array(obj):
        return topod(array2schema(obj))

    if is_schema(obj):
        return schema2pod(obj)

    if is_list(obj):
        return [topod(one) for one in obj]

    raise TypeError(f'can not convert to FR pod representation from {type(obj)}')

def toarray(obj):
    '''
    Return a array representation of an FR object.
    '''
    if is_array(obj):
        return obj

    if is_pod(obj):
        return toarray(pod2schema(obj))

    if is_schema(obj):
        return schema2array(obj);

    if is_list(obj):
        return [toarray(one) for one in obj]

    raise TypeError(f'can not convert to FR array representation from {type(obj)}')

# old function names
todict = topod
fromdict = pod2schema

def dumps(obj):
    '''
    Dump FR object in any rep to JSON text.
    '''
    return jsio.dumps(topod(obj), indent=2)


def loads(text):
    '''
    Load FR object from JSON text, return in schema representation.
    '''
    got = jsio.loads(text)
    if isinstance(got, dict):
        return pod2schema(got)
    return [pod2schema(one) for one in got]


def dump(path, obj, ext=""):
    '''Save an FR object.

    - path :: a file name or pathlib.Path object

    - ext :: a string like a file extension with which to judge format instead
      of considering the actual file name extension.  An extension of "npz"
      implies compression.  To force a .npz file with no compression pass
      ext="npy".
    '''
    dump_fr(obj, "dump")

    if isinstance(path, str):
        path = Path(path)

    print(f'dumping: {path=} {ext=}')

    if ext.endswith("npy"):
        dat = toarray(obj)
        numpy.savez(path, **dat)
        return

    if ext.endswith("npz") or path.suffix == ".npz":
        dat = toarray(obj)
        numpy.savez_compressed(path, **dat) # fixme: 
        return

    if ext.endswith("json") or path.suffix == ".json":
        text = dumps(obj)
        open(path, 'w').write(text)
        return

    if ext.endswith("json.bz2") or path.suffix == ".json.bz2":
        import bz2
        text = dumps(obj)
        bz2.BZ2File(path, 'wb').write(text.encode())
        return

    if ext.endswith("json.gz") or path.suffix == ".json.gz":
        import gzip
        text = dumps(obj)
        gzip.open(path, "wb").write(text.encode())
        return

    raise ValueError(f'unknown file format from path "{path}" and ext "{ext}"')


def load(path, ext=""):
    '''Return response.schema object or a list of them.

    - path :: a file name or pathlib.Path object or canonical detector name.

    - ext :: an extension by which to judge file format instead of using that from path.

    '''
    if isinstance(path, str):
        path = Path(path)

    print(f'loading: {path=} {ext=}')

    if ext.endswith(("npz", "npy")) or path.suffix == ".npz":
        got = dict(numpy.load(path))
        print(f'load {path} -> {type(got)}')
        got = array2schema(got)
        print(f'load {path} -> {type(got)}')
        return got

    if ext.endswith("json") or path.suffix == ".json":
        return loads(open(path, 'r').read())

    if ext.endswith("json.bz2") or path.suffix == ".json.bz2":
        import bz2
        return loads(bz2.BZ2File(path, 'r').read())

    if ext.endswith("json.gz") or path.suffix == ".json.gz":
        import gzip
        return loads(gzip.open(path, "rb").read())

    # path may be a detector name
    fields = detectors.load(path.name, "fields")
    if not fields:
        raise IOError(f'failed to load responses for detector "{path.name}"')

    if isinstance(fields, list):
        return [pod2schema(f) for f in fields]
    return pod2schema(fields)


