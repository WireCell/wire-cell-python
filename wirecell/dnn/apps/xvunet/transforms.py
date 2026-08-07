#!/usr/bin/env python
'''
The dataset transforms relevant to xvunet (same scheme as dnnroi_custom).
'''

from dataclasses import dataclass
from typing import Type
import torch                    # for float32 dtype


@dataclass
class DimParams:
    '''
    Per-dimension parameters for rec and tru dataset transforms.

    - crop :: a half-open range as slice
    - rebin :: an integer downsampling factor
    '''
    crop: slice
    rebin: int = 1

    def __post_init__(self):
        if not isinstance(self.crop, slice):
            self.crop = slice(*self.crop)


@dataclass
class Params:
    '''
    Common parameters for rec and tru dataset transforms.

    elech is for electronics channel dimension
    ticks is for sampling period dimension
    values are divided by norm
    '''
    elech: Type[DimParams]
    ticks: Type[DimParams]
    norm: float = 1.0


class Rec:
    '''
    The "rec" data transformation.
    '''

    default_params = Params(DimParams((0, 800), 1), DimParams((0, 1500), 1), 1.0)

    def __init__(self, params: Params = None, transpose: bool = False):
        self._params = params or self.default_params
        self.do_transpose = transpose

    def crop(self, x):
        return x[:, self._params.elech.crop, self._params.ticks.crop]

    def rebin(self, x):
        '''
        Average down by the per-dimension rebin factors, by splitting each
        axis into (n_out, factor) and taking the mean over the factor axes.
        The factors must divide the cropped shape exactly.
        '''
        ne, nt = self._params.elech.rebin, self._params.ticks.rebin,
        sh = (x.shape[0],                    # 0
              x.shape[1] // ne,              # 1
              ne,                            # 2
              x.shape[2] // nt,              # 3
              nt)                            # 4
        return x.reshape(sh).mean(4).mean(2) # (imgch, elech_rebinned, ticks_rebinned)

    def transform(self, x):
        if self.do_transpose:
            x = x.permute(0,2,1)
        x = self.crop(x)
        x = self.rebin(x)
        x = x/self._params.norm
        return x

    def __call__(self, x):
        '''
        Input and output are shaped:

        (# of image channels/layers, # electronic channels, # of time samples)

        Last two dimensions of output are rebinned.
        '''
        return self.transform(x)


class Tru(Rec):
    '''
    The "tru" data transformation: as "rec" but with thresholding to {0,1}.
    '''

    default_params = Params(DimParams((0, 800), 1), DimParams((0, 1500), 1), 0.05)

    def __init__(self, params: Params = None, transpose: bool = False,
                 threshold: float = 0.5):
        super().__init__(params or self.default_params, transpose)
        self.threshold = threshold

    def __call__(self, x):
        x = self.transform(x)
        return (x > self.threshold).to(torch.float32)
