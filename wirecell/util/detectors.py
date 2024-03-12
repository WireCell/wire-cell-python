from pathlib import Path
from .jsio import load as jload
from .paths import resolve as resolve_path

# This file is in wire-cell-data/ which should be in your WIRECELL_PATH.  A
# derived .json should also be found there but beware of it being older than the
# .jsonnet.
registry_file = "detectors.jsonnet"

def registry_path(registry = registry_file):
    return resolve_path(registry)    

# Resolve (detector name, file type) to a path or a list of paths.
def resolve(detname, regkey,
            registry = registry_file,
            paths=(), **kwds):
    '''
    Return file name or list of file names for the detname's regkey resolved via the file registry.
    '''
    registry = registry_path(registry)
    dets = jload(registry, paths, **kwds)
    det = dets[detname]
    fname = det[regkey]
    if isinstance(fname, str):
        return resolve_path(fname)
    return [resolve_path(fn) for fn in fname]
    

def load(name,
         regkey = None,
         registry = registry_file, 
         paths=(), **kwds):
    '''Load a JSON or Jsonnet "detector file".

    - name :: a direct file name to load() or a canonical detector name

    - registry :: a file name of a detector registry file 

    - regkey :: a file type name (eg "wires", "fields")

    - paths :: forwarded to jsio.load()

    - kwds :: forwarded to jsio.load()

    If name is a '.json' like file or regkey is None, act like jsio.load().
    Otherwise resolve name to file or files.  

    If resolving results in a list of file names then a list of corresponding
    loaded objects is returned.  Else, the object is returned.

    '''
    if '.json' in name or regkey is None: # act like load()
        return jload(name, paths, **kwds)

    fname = resolve(name, regkey, registry, paths, **kwds)

    if isinstance(fname,Path):
        return jload(fname, paths, **kwds);

    return [jload(fn, paths, **kwds) for fn in fname]

