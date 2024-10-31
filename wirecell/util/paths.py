#!/usr/bin/env python
'''
Deal with WCT related paths
'''
import os
from glob import glob
from pathlib import Path

def flatten(lst):
    return tuple([item for sublist in lst for item in sublist if item])


def listify(*lst, delim=":"):
    '''Return list.

      May give one or more lists, strings or Paths.  Strings are
      interpreted as possibly being delimited lists of strings.

    '''
    if not lst:
        return ()

    def listify_one(one):
        if not one:
            return ()
        if isinstance(one, Path):
            one = str(one)
        if isinstance(one, str):
            if delim in one:
                return flatten(map(listify_one, one.split(delim)))
            return (one,)
        return flatten(map(listify_one, one))

    return flatten(map(listify_one, lst))


def resolve(path, *pathlist):
    '''
    Return absolute path of path as a Path object

    If path is absolute, return path else search.

    The pathlist is first searched.  Then "." and then $WIRECELL_PATH.
    '''
    if isinstance(path, str):
        path = Path(path)
    
    if path.is_absolute():
        return path
    wcpath = os.environ.get("WIRECELL_PATH")
    pathlist = listify(pathlist, wcpath)
    for maybe in pathlist:
        maybe = Path(maybe) / path
        if maybe.exists():
            return maybe
    raise FileNotFoundError(f'failed to resolve {path.name}.  Check WIRECELL_PATH?')


def unglob(globs):
    '''
    Given a list of glob, return flat list of paths
    '''
    out = list()
    for one in globs:
        out += glob(one)
    return out
