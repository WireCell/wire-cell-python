#!/usr/bin/env python
'''
The dataset transforms relevant to DNNROI
'''

from dataclasses import dataclass
from typing import Type, Tuple
import torch                    # for float32 dtype


@dataclass
class DimParams:
    '''
    Per-dimension parameters for rec and tru dataset transforms.

    - crop :: a half-open range as slice
    - rebin :: an integer downsampling factor

    FYI, common channel counts per wire plane:
    - DUNE-VD:          256+320+288
    - DUNE-HD:          800+800+960
    - Uboone:           2400+2400+3456
    - SBND-v1:          1986+1986+1666
    - SBND-v0200:       1984+1984+1664
    - ICARUS:           1056+5600+5600
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
    The DNNROI "rec" data transformation.
    '''

    # default_params = Params(DimParams((476, 952), 1), DimParams((0,6000), 10), 4000)
    default_params = Params(DimParams((0, 1536), 1), DimParams((0,6000), 10), 4000)
    def __init__(self,  params: Params = None, transpose: bool = False):
        '''
        Arguments:

        - params :: a Params

        '''
        self._params = params or self.default_params
        self.do_transpose = transpose

    def crop(self, x):
        print('In crop, x:', x.shape)
        return x[:, self._params.elech.crop, self._params.ticks.crop]

    def rebin(self, x):
        ne, nt = self._params.elech.rebin, self._params.ticks.rebin,
        sh = (x.shape[0],                    # 0
              x.shape[1] // ne,              # 1
              ne,                            # 2
              x.shape[2] // nt,              # 3
              nt)                            # 4
        return x.reshape(sh).mean(4).mean(2) # (imgch, elech_rebinned, ticks_rebinned)
        
    def transform(self, x):
        if self.do_transpose:
            print('TRANSPOSING', x.shape)
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
    The DNNROI "tru" data transformation.

    This is same as "rec" but with a thresholding.
    '''

    # default_params = Params(DimParams((476, 952), 1), DimParams((0,6000), 10), 200)
    default_params = Params(DimParams((0, 1536), 1), DimParams((0,6000), 10), 200)
    def __init__(self, params: Params = None,  transpose : bool = False, threshold: float = 0.5):
        '''
        Arguments (see Rec for more):

        - threshold :: threshold for array values to be set to 0 or 1.
        '''
        super().__init__(params=(params or self.default_params), transpose=transpose)
        self.threshold = threshold
        
    def __call__(self, x):
        x = self.transform(x)
        return (x > self.threshold).to(torch.float32)


