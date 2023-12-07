#!/usr/bin/env python3
'''
Uniform wrapper over json/jsonnet loading
'''

import os
import bz2
import json
import gzip
from pathlib import Path

def jsonnet_module():
    try:
        import _gojsonnet as _jsonnet
    except ImportError:
        import _jsonnet
    return _jsonnet

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
    '''Resolve filename against built-in directories and any
    user-provided list in "paths".

    Raise ValueError if fail.

    '''
    if not filename:
        raise RuntimeError("no file name provided")
    if isinstance(filename, str):
        filename = Path(filename)
    if filename.is_absolute():
        return filename

    for maybe in clean_paths(paths):
        maybe = Path(maybe) / filename
        if maybe.exists():
            return maybe
    raise RuntimeError(f"file name {filename} not resolved in paths {paths}")


def try_path(path, rel):
    '''
    Try to open a path
    '''
    path = Path(path)
    rel = Path(rel)

    if rel.is_absolute():
        full_path = rel
    else:
        fulll_path = path / rel

    if full_path.is_dir():
        raise RuntimeError('Attempted to import a directory')

    if full_path.exists():
        return full_path, None
    # https://github.com/google/jsonnet/releases/tag/v0.19.1
    jsmod = jsonnet_module()
    import semver
    import_returns_bytes = semver.compare(getattr(jsmod, 'version', 'v0.18.0')[1:], '0.18.0') > 0
    flags = 'r'
    if import_returns_bytes:
        flags = 'rb'
    return full_path, full_path.read()


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
        raise RuntimeError('file not found for import')

def file_object(fname, opt='r'):
    '''
    Return an open file object.

    A decompressing file object is returned if so indicated by the
    file name extension.
    '''
    fname = Path(fname)
    
    if fname.suffix == ".gz":
        return gzip.open(fname, opt)
    if fname.suffix == ".bz2":
        return bz2.open(fname, opt)
    return open(fname, opt)


def loads(text):
    '''
    Load object from JSON text.
    '''
    return fromdict(json.loads(text))


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

    if fname.name.endswith(('.jsonnet', '.jsonnet.gz', '.jsonnet.bz2')):
        ic = ImportCallback(paths)
        jsmod = jsonnet_module()
        try:
            text = jsmod.evaluate_snippet(str(fname.absolute()), text, import_callback=ic, **kwds)
        except RuntimeError as err:
            raise RuntimeError(f"in file: {fname}") from err
    elif fname.name.endswith(('.json', '.json.bz2', '.json.gz')):
        pass
    else:
        raise RuntimeError(f'unsupported file extension {fname}')
    return json.loads(text)


def dumps(obj, indent=2):
    '''
    Dump object to JSON text.
    '''
    return json.dumps(todict(obj), indent=indent)


def dump(f, obj, index=2):
    '''
    Save object obj to file name or file object of f.
    '''
    btext = dumps(obj, indent=indent).encode()
    f = Path(f)

    if isinstance(f, str):
        if f.name.endswith(".json"):
            open(f, 'wb').write(btext)
            return
        if f.name.endswith(".json.bz2"):
            import bz2
            bz2.BZ2File(f, 'w').write(btext)
            return
        if f.name.endswith(".json.gz"):
            import gzip
            gzip.open(f, "wb").write(btext)
            return
        raise RuntimeError("unknown file format: %s" % filename)
    f.write(btext);


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
    except RuntimeError:
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

    # Path'ify with possible split-on-:
    paths = list()
    for one in path:
        if isinstance(one, Path):
            paths.append(one)
        else:
            paths += [Path(p) for p in one.split(":")]

    return [p for p in paths if
            p.is_dir() and p.exists()]
