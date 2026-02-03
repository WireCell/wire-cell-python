#!/usr/bin/env python
'''
Datasets for "frame" files.

Instead of importing directly from here, consider using the class names provided
indirectly by dnn.data.
'''

# A sample is defined by a list of regex which must include one group giving their common ID.

# m = re.match(r'/(\d{3})/(?:frame_ductor0|frame_blah0)', '/102/frame_ductor0')
# m[1] -> '102'

import re
import h5py
from collections import defaultdict
import torch
from torch.utils.data import Dataset

from dataclasses import dataclass
import logging
log = logging.getLogger("wirecell.dnn")

def allkeys(obj):
    "Recursively find all keys in an h5py.Group."
    keys = (obj.name,)

    if isinstance(obj, h5py.Group):
        for key, value in obj.items():
            if isinstance(value, h5py.Group):
                keys = keys + allkeys(value)
            else:
                keys = keys + (value.name,)
    return keys
 

class ReMatcher:
    '''
    A callable returning a triple:

    
    (file ID, sample-in-file ID, layer-in-sample ID)

    or none given a file name and file key.

    The layer IDs are determined from a list of regular expressions to match
    against the path (file key).

    All regex must include matcing one regex "group" to provide the sample ID.
    The layer ID is taken as the index of the matching regex in the relist.
    '''

    def __init__(self, file_re, layer_relist):
        self.file_re = re.compile(file_re)
        self.layer_res = list(map(re.compile, layer_relist))
        self._fids = dict()
        
    def __call__(self, fname, fkey):
        fid = self._fids.get(fname, None)
        if fid is None:
            print(f'Rematch {self.file_re} vs {fname} ')
            print(self.file_re)
            m = self.file_re.match(fname)
            if not m:
                print('here')
                return
            fid = self._fids[fname] = m[1]
        
        for lid, rec in enumerate(self.layer_res):
            m = rec.match(fkey)
            if not m:
                continue
            sid = m[1]
            return fid,sid,lid
        
    def __len__(self):
        return len(self.layer_res)


class Domain:
    '''
    Configure how the array for a domain is produced and managed.

    Provide a matcher to return triple: (file, sample, layer) ID strings given
    a file and an HDF5 dataset path name.

    If a transform is given it is called on the array and the result is
    returned.

    Set cache to True to retain the transformed array in memory to avoid
    reloading if subsequently accessed.  Default is False.  When caching, the
    array is provided via a .detach(). 

    Set grad to True to cause gradients to be calculated and attached.  Default
    is False.
    '''

    def __init__(self, matcher, transform=None, cache=False, grad=False, name=""):
        self.match=matcher
        self._transform=transform
        self.cache=cache
        self.grad=grad
        self.name=name
        self.preload = True     # preload cache (if cache=True)

    def transform(self, x):
        if self._transform:
            x = self._transform(x)
        return x

    def __str__(self):
        return str(self.__dict__)

class Single(Dataset):
    '''
    A dataset yielding a single array (not in a tuple).
    '''

    def __init__(self, domain, paths=(), dtype=torch.float32):
        self.domain = domain
        self._index = list()  # idx->[(fp,fkey), ...]
        self._cache = dict()  # idx->arr1
        self._dtype = dtype
        if paths:
            self.append(paths)
        if domain.cache and domain.preload:
            n = len(self)
            log.debug(f'hdf.Single preloading {n} in {domain.name}')
            for idx in range(n):
                self[idx]
        log.debug(f'hdf.Single {domain.name}')
        
    def append(self, paths):
        '''
        Scan files to build index to individual layer arrays.
        '''

        byids = defaultdict(dict)  # (fid,sid) -> (lid) -> (fp,fkey)
        for fname in paths:
            fp = h5py.File(fname)
            log.debug(f'scanning {fp.filename}')
            for fkey in allkeys(fp):
                print(fkey)
                val = fp.get(fkey)
                if not isinstance(val, h5py.Dataset):
                    continue
                print('is instance. Checking ', fname, fkey)
                got = self.domain.match(fname, fkey)
                print(got)
                if not got:
                    # print(f'{fname}:{fkey} not in {self.domain.name}')
                    continue
                fid, sid, lid = got
                kid = (fid,sid)
                byids[kid][lid] = (fp, fkey)
                print('good', fkey)

        for kid in sorted(byids):
            entry = byids[kid]
            layers = [entry[lid] for lid in sorted(entry)]
            if len(layers) != len(self.domain.match):
                raise ValueError('failed to find correct number of layers: '
                                 f'{len(layers)} != {len(self.domain.match)} for layer {layers}')
            self._index.append(layers)

    def _load_entry(self, idx):
        '''
        Load, build and transform array from file.  
        '''
        # log.debug(f'hdf loading: [{idx}]')
        layers = self._index[idx]

        tens = list()
        keys = list()
        for fp, key in layers:
            # log.debug(f'{fp} {key}')
            # this takes about 75% of the time
            d = fp.get(key)
            keys.append(key)
            # takes about 5%
            ten = torch.tensor(d[:], requires_grad = False).to(dtype=self._dtype)
            log.debug(f'Appending {ten.shape}')
            tens.append(ten)
        # this about 15% 
        ten = torch.stack(tens, axis=2)  # (ntick, nchan, nlayer)
        ten = torch.transpose(ten, 0, 2)   # (nlayer, nchan, ntick)

        ten = self.domain.transform(ten)
        log.debug(f'hdf loaded [{idx}]: {ten.shape} {keys} in {fp.filename}')

        # Delay requiring gradients as stack() does not and arbitrary
        # transform() may not have gradients.  Not doing this may also be
        # related to error "Trying to backward through the graph a second time".
        ten.requires_grad = self.domain.grad

        return ten

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        ten = self._cache.get(idx, None)  # cache or not, if we have it, we have it
        if ten is not None:
            # print(f'hdf from cache [{idx}]: {ten.shape} {ten.dtype} {ten.device} {self.domain.name}')            
            return ten

        ten = self._load_entry(idx)

        if self.domain.cache:
            self._cache[idx] = ten

        return ten


class Multi(Dataset):
    '''
    Effectively a zip(*singles).

    The sample at an index is a tuple of arrays returned from each singles dataset at that index.
    '''
    def __init__(self, *singles):
        if len(set([len(one) for one in singles])) != 1:
            #print(singles)
            raise ValueError('different size singles')
        self._singles = singles

    def __len__(self):
        if len(self._singles) == 0:
            return 0
        return len(self._singles[0])

    def __getitem__(self, idx):
        return tuple([one[idx] for one in self._singles])
