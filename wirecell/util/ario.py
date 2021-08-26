#!/usr/bin/env python3
'''
Array / archive IO

This module provides a numpy.load() like dict'ish which can handle
more than just .npz/zip archives and supports archives holding more
than .npy files.

'''
import os
import io
import json
import numpy
import functools
import zipfile
import tarfile

def npz_load(filename):
    '''
    Load an npz file
    '''
    return numpy.load(filename)

# XxxReader constructs on a file name and provides items as bytes and
# list of archive names as keys().

class TarReader(dict):

    exts = ('.tar', '.tar.bz2', '.tar.gz', '.tar.xz')

    def __init__(self, filename):
        mode = "r"
        for maybe_compressed in ('bz2','gz','xz'):
            if filename.endswith(".tar."+maybe_compressed):
                mode += ':' + maybe_compressed
        tf = tarfile.open(filename, mode)
        for ti in tf.getmembers():
            #print(f'load from {filename}: {ti.name}')
            self[ti.name] = tf.extractfile(ti).read()

    
    # def __getitem__(self, name):
    #     return self.

    # @functools.cache
    # def keys(self):
    #     return [ti.name for ti in self.tf.getmembers()]


class ZipReader(dict):

    exts = ('.zip', '.npz')

    def __init__(self, filename):
        self.zf = zipfile.ZipFile(filename)
        for name in self.zf.namelist():
            self[name] = self.zf.open(name).read()

def reader(filename):
    for R in (TarReader, ZipReader):
        for ext in R.exts:
            if filename.endswith(ext):
                return R(filename)
    raise ValueError(f'no reader for {filename}')
    

class Reader(dict):
    '''
    Mimic a numpy.load() dict'ish for various archve formats
    '''

    # Known archive file member object formats and their resolution
    # order when guessing.
    formats = ("npy", "json")

    def __init__(self, filename, keep_extension=False):
        '''
        Construct a reader on an archive file name.

        If keep_extension is True then archive object is retrieved via
        its archive file name including extension and otherwise just
        by its basename (latter is default to match numpy.load()
        behavior).
        '''
        r = reader(filename)
        for fname, dat in r.items():
            #print (f'Reader: {fname}')

            key, ext = os.path.splitext(fname)
            if ext == ".json":
                obj = json.loads(dat.decode())
            elif ext == ".npy":
                a = io.BytesIO()
                a.write(dat)
                a.seek(0)
                obj = numpy.load(a)
                #print(f'{key}: {obj.shape} {obj.dtype}')
            else:
                obj = dat       # as-is
            if keep_extension:
                key = fname
            self[key] = obj
                
def load(filename):
    '''
    Load a file into a dict of fledged objects.
    '''
    return Reader(filename)
