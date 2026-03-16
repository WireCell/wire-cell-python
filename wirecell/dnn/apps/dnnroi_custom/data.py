#!/usr/bin/env python
'''
Dataset and model specific to DNNROI training.

See for example:

https://www.phy.bnl.gov/~hyu/dunefd/dnn-roi-pdvd/Pytorch-UNet/data/
'''
# fixme: add support to load from URL

from wirecell.dnn.data import hdf

from .transforms import Rec as Rect, Tru as Trut, Params as TrParams, DimParams

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
            'frame_loose_lf']
    )

    def __init__(self, paths, 
                 file_re=None, path_res=None,
                 trparams: TrParams = Trut.default_params, cache=False):
        print(file_re, self.file_re)
        print(path_res, self.path_res)
        dom = hdf.Domain(hdf.ReMatcher(file_re or self.file_re,
                                       path_res or self.path_res),
                         transform=Rect(trparams, transpose=True),
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
        r'/(\d+)/%s\d'%tag for tag in ['frame_deposplat']
    )

    def __init__(self, paths, threshold = 0.5,
                 file_re=None, path_res=None,
                 trparams: TrParams = Trut.default_params, cache=False):

        dom = hdf.Domain(hdf.ReMatcher(file_re or self.file_re,
                                       path_res or self.path_res),
                         transform=Trut(trparams, True, threshold),
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

        tru_norm = wash('tru_norm')
        tru_tick_binning = wash('tru_tick_binning')
        tru_elech_binning = wash('tru_elech_binning')
        
        rec_norm = wash('rec_norm')
        rec_tick_binning = wash('rec_tick_binning')
        rec_elech_binning = wash('rec_elech_binning')
        
        if ((tru_norm is None) or (tru_tick_binning is None) or (tru_elech_binning is None)):
            tru_params = Trut.default_params
        else:
            tru_elech_binning = [int(i) for i in tru_elech_binning]
            tru_tick_binning = [int(i) for i in tru_tick_binning]
            tru_params = TrParams(
                DimParams(tru_elech_binning[0:2], tru_elech_binning[2]),
                DimParams(tru_tick_binning[0:2], tru_tick_binning[2]),
                float(tru_norm)
            )
        if ((rec_norm is None) or (rec_tick_binning is None) or (rec_elech_binning is None)):
            rec_params = Trut.default_params
        else:
            rec_elech_binning = [int(i) for i in rec_elech_binning]
            rec_tick_binning = [int(i) for i in rec_tick_binning]
            rec_params = TrParams(
                DimParams(rec_elech_binning[0:2], rec_elech_binning[2]),
                DimParams(rec_tick_binning[0:2], rec_tick_binning[2]),
                float(rec_norm)
            )

        print('Tru params:', tru_params)
        print('Rec params:', rec_params)
        super().__init__(Rec(paths, cache=cache,
                             trparams=rec_params,
                             file_re=wash('rec_file_re'),
                             path_res=wash('rec_path_res')),
                         Tru(paths, threshold, cache=cache,
                             trparams=tru_params,
                             file_re=wash('tru_file_re'),
                             path_res=wash('tru_path_res')))