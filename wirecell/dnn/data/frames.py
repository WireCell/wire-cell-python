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
 
class H5(Dataset):
    '''
    A dataset for HDF5 "frame" files
    '''

    re_dnnroi_rec = tuple(
        r'/(\d+)/%s\d'%tag for tag in ['frame_loose_lf', 'frame_mp2_roi', 'frame_mp3_roi']
    )
    re_dnnroi_tru = tuple(
        r'/(\d+)/%s\d'%tag for tag in ['frame_ductor']
    )
    

    def __init__(self, paths, matchers, transform=None):
        '''
        Create an frames.H5 dataset.

        - paths gives a list of HDF5 file names to use.
        - matchers is a list of regex.

        Each regex in the list is used to define one layer in the image channel
        dimension and the list order determines order on this dimension.  Each
        regex must have one group which is used to define the ID for which image
        channels are grouped.  A failed re.match() is an error.

        Note, IDs are matched only in the context of a single file.

        Common regular expression lists are given as class members re_*.
        '''
        self.transform = transform

        # (fileobject, ID) -> [(layer,key)]
        index = defaultdict(list)

        for path in paths:
            fp = h5py.File(path)
            for key in allkeys(fp):
                val = fp.get(key)
                if not isinstance(val, h5py.Dataset):
                    continue
                for layer, pattern in enumerate(matchers):
                    m = re.match(pattern, key)
                    if m:
                        reid = m[1]
                        index[(fp,reid)].append((layer, key))
                    
        self._entries = list()
        for (fp,ID),layers in index.items():
            if len(layers) != len(matchers):
                raise ValueError(f'Failed to find layers: {len(layers)} != {len(matchers)}')
            layers.sort()
            self._entries.append((fp, [l[1] for l in layers]))

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, idx):
        fp, keys = self._entries[idx]
        print (fp, keys)

        layers = list()
        for key in keys:
            d = fp.get(key)     # this takes about 75% of the time
            layers.append( torch.tensor(d[:]) )  # about 5%
        return torch.stack(layers, axis=2)  # this about 15%
