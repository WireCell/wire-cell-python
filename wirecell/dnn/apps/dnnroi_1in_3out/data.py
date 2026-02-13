#!/usr/bin/env python
'''
Dataset and model specific to DNNROI training.

See for example:

https://www.phy.bnl.gov/~hyu/dunefd/dnn-roi-pdvd/Pytorch-UNet/data/
'''
# fixme: add support to load from URL

from wirecell.dnn.data import hdf

from .transforms import Rec as Rect, Tru as Trut, Params as TrParams

import logging
log = logging.getLogger("wirecell.dnn")

class Rec(hdf.Single):
    '''
    A DNNROI "rec" dataset.

    This consists of conventional sigproc results produced by WCT's
    OmnibusSigProc in HDF5 "frame file" form.
    '''

    file_re = r'.*g4-rec-[r]?(\d+)\.h5'

    path_res = tuple(
        r'/(\d+)/%s\d'%tag for tag in [
            'frame_loose_lf',# 'frame_mp2_roi', 'frame_mp3_roi'
        ]
    )

    def __init__(self, paths, 
                 file_re=None, path_res=None,
                 trparams: TrParams = Trut.default_params, cache=False):

        dom = hdf.Domain(hdf.ReMatcher(file_re or self.file_re,
                                       path_res or self.path_res),
                         transform=Rect(trparams),
                         cache=cache, grad=True,
                         name="dnnroirec")
        super().__init__(dom, paths)


class Tru(hdf.Single):
    '''
    A DNNROI "tru" dataset.

    This consists of the target ROI
    '''

    file_re = r'.*g4-tru-[r]?(\d+)\.h5'

    path_res = tuple(
        r'/(\d+)/%s\d'%tag for tag in [
            'frame_deposplat', 'frame_mp3', 'frame_mp2'
        ]
    )

    def __init__(self, paths, threshold = 0.5,
                 file_re=None, path_res=None,
                 trparams: TrParams = Trut.default_params, cache=False):

        dom = hdf.Domain(hdf.ReMatcher(file_re or self.file_re,
                                       path_res or self.path_res),
                         transform=Trut(trparams, threshold),
                         cache=cache, grad=False,
                         name="dnnroitru")

        super().__init__(dom, paths)




class Dataset(hdf.Multi):
    '''
    The full DNNROI dataset is effectively zip(Rec,Tru).
    '''
    def __init__(self, paths, threshold=0.5, cache=False, config=None):

        log.debug(f'ddnroi dataset: {config=}')
        config = config or dict()
        def wash(key):
            val = config.get(key, None)
            if val is None:
                return 
            if isinstance(val, str) and val.startswith(('[','{')):
                val = eval(val)         # yes, I know
                log.debug(f'dnnroi dataset {key} = {val}')
            return val


        # fixme: allow configuring the transforms.
        super().__init__(Rec(paths, cache=cache,
                             file_re=wash('rec_file_re'),
                             path_res=wash('rec_path_res')),
                         Tru(paths, threshold, cache=cache,
                             file_re=wash('tru_file_re'),
                             path_res=wash('tru_path_res')))

