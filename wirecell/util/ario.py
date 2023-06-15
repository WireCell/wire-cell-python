#!/usr/bin/env python3
'''Array / archive IO

Read-only dict-like, sometimes efficient, random access to files.

'''
import io
import json
import numpy
import pathlib
import zipfile
import tarfile
from collections.abc import Mapping

def stem_if(fname, exts):
    '''
    Return stem if extension in exts
    '''
    stem, ext = fname.rsplit(".", 1)
    if ext in exts:
        return stem
    return fname

import gzip
import bz2
import lzma
decompressors = dict(
    gz = gzip.decompress,
    bz2 = bz2.decompress,
    xz = lzma.decompress
)


def transform(fname, data):
    '''
    Transform data from filename to object.
    '''
    fname, ext = fname.rsplit(".", 1)
    if ext in decompressors:
        data = decompressors[ext](data)
        fname, ext = fname.rsplit(".", 1)
    if ext.lower() in ("json",):
        return json.loads(data)
    if ext.lower() in ("npy","numpy"):
        bio = io.BytesIO()
        bio.write(data)
        bio.seek(0)
        return numpy.load(bio)
    raise ValueError(f'unsupported member type: {ext}')


class Arf(Mapping):
    '''
    Base for common methods in Tar/Zip.

    Subclass provides self._index a dictionary.  If self._lazy is
    true, indexed values are called and their returns are returned.
    Else, indexed objects are returned directly.
    '''

    def lazy_load(self, fileobj, infoobj):
        '''
        Return a callable that loads an object given a file and info
        object.
        '''
        def loader():
            return self.greedy_load(fileobj, infoobj)
        return loader

    def __getitem__(self, key):
        val = self._index[key]
        if self._lazy:
            return val()
        return val

    def __iter__(self):
        return iter(self._index)

    def __len__(self):
        return len(self._index)

        

class Tar(Arf):
    '''
    Apply a mapping interface to a tar file
    '''

    def __init__(self, path, lazy=True):
        '''
        Create a Tar mapping on file path.  
        '''
        self.path = path
        mode = "r"
        if path.endswith(('.gz', '.tgz')):
            mode += ':gz'
            lazy = False
        elif path.endswith(('.xz', '.txz')):
            mode += ':xz'
            lazy = False
        elif path.endswith(('.bz2', '.tbz2')):
            mode += ':bz2'
            lazy = False
        tf = tarfile.open(path, mode)
        self._lazy = lazy
        if lazy:
            tran = self.lazy_load
        else:
            tran = self.greedy_load
        self._index = dict()
        self.member_names = dict()
        for ti in tf.getmembers():
            key = self.keymap(ti)
            if not key:
                continue
            self._index[key] = tran(tf, ti)
            self.member_names[key] = ti.name


    def keymap(self, ti):
        '''
        Return lookup key.
        '''
        if not ti.isfile():
            return
        # strip internal compression extension
        key = stem_if(ti.name, ('gz', 'xz', 'bz2'))
        # strip of object extension
        key = stem_if(key, ('npy', 'json'))
        return key

    def greedy_load(self, tf, ti):
        '''
        Immediately load the TarInfo ti from TarFile tf and return
        object.
        '''
        data = tf.extractfile(ti).read()
        return transform(ti.name, data)


class Zip(Arf):
    '''
    Apply a mapping interface to a zip file.

    This will lazy load.
    '''
    def __init__(self, path, lazy=True):
        self.path = path
        zf = zipfile.ZipFile(path)
        self._lazy = lazy
        if lazy:
            tran = self.lazy_load
        else:
            tran = self.greedy_load
        self._index = dict()
        self.member_names = dict()
        for zi in zf.infolist():
            key = self.keymap(zi)
            if not key:
                continue
            self._index[key] = tran(zf, zi)
            self.member_names[key] = zi.filename

    def keymap(self, zi):
        '''
        Return lookup key.
        '''
        if zi.is_dir():
            return
        # strip internal compression extension
        key = stem_if(zi.filename, ('gz', 'xz', 'bz2'))
        # strip of object extension
        key = stem_if(key, ('npy', 'json'))
        return key

    def greedy_load(self, zf, zi):
        '''
        Immediately load the ZipInfo zi from ZipFile zf and return
        object.
        '''
        data = zf.open(zi.filename).read()
        return transform(zi.filename, data)


class Pixz:
    # reminder to make this one day
    pass


def reader_class(path):
    '''
    Make a reader class based on parsing a path
    '''
    path = pathlib.Path(path)
    if not path.exists():
        raise ValueError(f'no such file: {path.name}')
    if path.name.endswith(('.tar', '.tar.gz', '.tar.bz2', '.tar.xz', '.tgz', '.txz', '.tbz2')):
        return Tar
    if path.name.endswith(('.zip', '.npz')):
        return Zip
    if path.name.endswith(('.tix', '.tar.pixz', '.tpxz')):
        return Pixz

    raise ValueError(f'unsupported archive type: {path.name}')    
    

def load(path, lazy=True):
    '''
    Load a file into a dict of fledged objects.

    If lazy, delay loading until a key is accessed.  Compressed tar
    archives which lack indices (eg, non-pixz) can not be lazy loaded
    and will always be greedy-loaded.  If application intends to load
    the entire archive then greedy-loading it faster.
    '''
    Reader = reader_class(path)
    ret = Reader(path, lazy)
    #print(list(ret.keys()))
    return ret
