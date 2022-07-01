#!/usr/bin/env python3
'''
Uniform wrapper over json/jsonnet loading
'''

import os
import bz2
import json
import gzip
from _jsonnet import evaluate_file, evaluate_snippet
from pathlib import Path

def clean_paths(paths, add_cwd=True):
    '''Return list of paths made absolute with cwd as first .

    Input "paths" may be a ":"-separated string or list of string.

    If add_cwd is True and if cwd is not already in paths, it will be
    prepended.

    '''
    if isinstance(paths, str):
        paths = paths.split(":")
    paths = [os.path.realpath(p) for p in paths]

    if add_cwd:
        cwd = os.path.realpath(os.path.curdir)
        if cwd not in paths:
            paths.insert(0, cwd)

    return paths


def resolve(filename, paths=()):
    '''Resolve filename against moo built-in directories and any
    user-provided list in "paths".

    Raise ValueError if fail.

    '''
    if not filename:
        raise ValueError("no file name provided")
    if filename.startswith('/'):
        return filename

    for maybe in clean_paths(paths):
        fp = os.path.join(maybe, filename)
        if os.path.exists(fp):
            return fp
    raise ValueError(f"file not found: {filename}")

def try_path(path, rel):
    '''
    Try to open a path
    '''
    if not rel:
        raise RuntimeError('Got invalid filename (empty string).')
    if rel[0] == '/':
        full_path = rel
    else:
        full_path = os.path.join(path, rel)
    if full_path[-1] == '/':
        raise RuntimeError('Attempted to import a directory')

    if not os.path.isfile(full_path):
        return full_path, None
    with open(full_path) as f:
        return full_path, f.read()

class ImportCallback(object):

    def __init__(self, paths=()):
        self.paths = list(paths)
        self.found = set()

    def __call__(self, path, rel):
        paths = [path] + self.paths
        for maybe in paths:
            try:
                full_path, content = try_path(maybe, rel)
            except RuntimeError:
                continue
            if content:
                self.found.add(full_path)
                return full_path, content
        raise RuntimeError('File not found')

def file_object(fname, opt='r'):
    '''
    Return an open file object.

    A decompressing file object is returned if so indicated by the
    file name extension.
    '''
    if fname.endswith(".gz"):
        return gzip.open(fname, opt)
    if fname.endswith(".bz2"):
        return bz2.open(fname, opt)
    return open(fname, opt)


def load(fname, paths=(), **kwds):
    '''
    Load JSON or Jsonnet file, returning data.

    Format is guessed from file name extensions.

    Compression extenstions .gz and .bz2 supported.

    See https://jsonnet.org/ref/bindings.html for list of kwds known
    to the Jsonnet loader.
    '''

    paths = clean_paths(paths)
    fname = resolve(fname, paths)

    fp = file_object(fname, 'rb')
    text = fp.read().decode()

    if fname.endswith(('.jsonnet', '.jsonnet.gz', '.jsonnet.bz2')):
        ic = ImportCallback(paths)
        try:
            text = evaluate_snippet(fname, text, import_callback=ic, **kwds)
        except RuntimeError as err:
            raise RuntimeError(f"in file: {fname}") from err
    elif fname.endswith(('.json', '.json.bz2', '.json.gz')):
        pass
    else:
        raise RuntimeError(f'unsupported file extension {fname}')
    return json.loads(text)


def scalar_typify(val):
    '''
    Return tuple (value, iscode)

    If iscode is true if value should be considered for tla_codes.

    The value is turned into a string.
    '''
    if not isinstance(val, str):
        return (str(val), True)
    try:
        junk = float(val)
        return (val, True)
    except ValueError:
        pass
    if val.lower() in ("true", "yes", "on"):
        return ("true", True)
    if val.lower() in ("false", "no", "off"):
        return ("false", True)
    return (val, False)

def tla_pack(tlas, paths=(), pre='tla_'):
    '''
    Convert strings in form key=val, key=code or key=filename into
    kwds ready to give to jsonnet.evaluate_file().

    The paths list of directories will be searched to dtermine if a
    key provides a filename.

    This function can be used for ext vars by passing pre="ext_".
    '''
    vars = dict()
    codes = dict()

    for one in tlas:
        try:
            key,val = one.split("=",1)
        except ValueError as err:
            raise ValueError("Did you forget to specify the TLA variable?") from err
        
        # Does it look like code?
        if val[0] in '{["]}':
            codes[key] = val
            continue

        # Maybe a file?
        chunks = val.split(".")
        if len(chunks) > 1:
            ext = chunks[-1]
            if ext in [".jsonnet", ".json", ".schema"]:
                fname = resolve(val, paths)
                codes[key] = load(fname, paths)
                continue

        # Must be some scalar value
        val, iscode = scalar_typify(val)
        if iscode:
            codes[key] = val
        else:
            vars[key] = val

    # these keywords are what jsonnet.evaluate_file() expects
    return {pre+'vars':vars, pre+'codes': codes}
        
def wash_path(path):
    '''
    Given one or more strings that are directory paths or :-separated
    lists of such, return a flat list of paths, in order, that
    actually are directories and exist.
    '''
    if isinstance(path, str):
        path = [path]
    parts = list()
    for one in path:
        parts += one.split(":")
    ret = list()
    for one in parts:
        p = Path(one)
        if not p.exists():
            print(f'skipping missing directory: {one}')
            continue
        if not p.is_dir():
            print(f'skipping non-directory: {one}')
            continue
        ret.append(str(p.absolute()))
    return ret
