#!/usr/bin/env python3
'''
Some utility functions for dealing with file I/O
'''
import os
import zipfile
import tarfile
from pathlib import Path

# fixme: more generic path functions are in jsio which should move here
def wirecell_path():
    '''
    Return list of paths from WIRECELL_PATH.
    '''
    return tuple(os.environ.get("WIRECELL_PATH","").split(":"))

def source_type(name):
    '''
    Return a canonical type for a source which matches a *ball() function.
    '''
    path = Path(name)
    if path.is_dir():
        return "dir"

    sufs = path.suffixes
    if not sufs:
        return "dat"
    if sufs[-1] == ".zip":
        return "zip"
    if sufs[-1] in (".tgz", ".tar") or name.endswith(".tar.gz"):
        return "tar"
    return "dat"
    

def maybe_decode(dat, decode=True, **kwds):
    if not decode:
        return dat
    if isinstance(decode, str):
        return dat.decode(encoding)
    return dat.decode()

def datball(filename, **kwds):
    '''
    A source which is a single data file to read directly.
    '''
    path = Path(filename)
    if not path.exists():
        raise ValueError(f'no such datfile {filename}')
    yield filename, maybe_decode(path.read_bytes(), **kwds)


def zipball(filename, **kwds):
    '''
    A zip file source
    '''
    path = Path(filename)
    if not path.exists():
        raise ValueError(f'no such zipfile {filename}')

    zf = zipfile.ZipFile(filename, 'r')
    for fname in zf.namelist():
        with zf.open(fname) as fp:
            yield fname, maybe_decode(fp.read(), **kwds)


def tarball(filename, **kwds):
    '''
    A tar file source.
    '''
    path = Path(filename)
    if not path.exists():
        raise ValueError(f'no such tarfile {filename}')

    tf = tarfile.open(filename, 'r')
    for name,member in sorted([(m.name,m) for m in tf.getmembers()]):
        if member.isdir():
            continue
        yield member.name, maybe_decode(tf.extractfile(member).read(), **kwds)
    

def dirball(dirname, pattern="*.*", **kwds):
    '''
    A source in the form of a directory.

    Special kwd args:

        - pattern :: a glob pattern to generate which files to
          consider.  If pattern is a list, glob on each element.

    '''
    pdir = Path(dirname)
    if not pdir.exists():
        raise ValueError(f'not such directory {dirname}')

    if isinstance(pattern, str):
        pattern = [pattern]

    paths = list()
    for p in pattern:
        paths += pdir.glob(p)

    for path in paths:
        for one in load(str(path), pattern=pattern, **kwds):
            yield one


def load(name, **kwds):
    '''
    Return generator of sequence of 2-tuple

    (file name, file contents)

    Contents are bytes.

    See source_type() for supported types.
    '''
    typ = source_type(name)
    here = globals()
    methname = f'{typ}ball'
    meth = here.get(methname, None)
    if not meth:
        raise ValueError(f'unsupported source type: {name}')
    return meth(name, **kwds)

