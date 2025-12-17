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

    path_res = tuple(
        r'/(\d+)/%s\d'%tag for tag in [
            'frame_loose_lf',
            # 'frame_decon_charge',
            # 'frame_gauss'
        ]
    )

    def __init__(self, paths, 
                 file_re=None, path_res=None,
                 trparams: TrParams = Trut.default_params, cache=False,
                 transpose=False,
                 det='hd',
                ):

        file_re = r'.*g4-rec-[r]?(\d+)\.h5' if det == 'vd' else r'.*g4-rec-[r]?(\d{8}T\d{6}Z)\.h5'
        dom = hdf.Domain(hdf.ReMatcher(file_re or self.file_re,
                                       path_res or self.path_res),
                         transform=Rect(params=trparams, transpose=transpose),
                         cache=cache, grad=True,
                         name="dnnroirec")
        super().__init__(dom, paths)


class Tru(hdf.Single):
    '''
    A DNNROI "tru" dataset.

    This consists of the target ROI
    '''


    file_re = r'.*g4-tru-[r]?(\d+)\.h5'
    # path_res = tuple(
    #     r'/(\d+)/%s\d'%tag for tag in ['frame_ductor']
    # )

    # file_re = r'.*g4-tru-[r]?(\d{8}T\d{6}Z)\_cleaned\.h5'
    # path_res = tuple(
    #     r'/(\d+)/%s\d'%tag for tag in ['frame_deposplat']
    # )
    def __init__(self, paths, threshold = 0.5,
                 file_re=None, path_res=None,
                 trparams: TrParams = Trut.default_params, cache=False,
                 transpose=False, det='hd',
                ):
        file_re = r'.*g4-tru-[r]?(\d+)\.h5' if det == 'vd' else r'.*g4-tru-[r]?(\d{8}T\d{6}Z)\_cleaned\.h5'
        path_res = tuple(
            r'/(\d+)/%s\d'%tag for tag in ['frame_ductor' if det=='vd' else 'frame_deposplat']
        )
        dom = hdf.Domain(hdf.ReMatcher(file_re or self.file_re,
                                       path_res or self.path_res),
                         transform=Trut(params=trparams, transpose=transpose, threshold=threshold),
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

        if ('wire_transform' in config.keys() and
            'tick_transform' in config.keys() and
            config['wire_transform'] != '' and
            config['tick_transform'] != ''):
            wt = [int(i) for i in config['wire_transform'].split()]
            tt = [int(i) for i in config['tick_transform'].split()]
            print('wire_transform', wt)
            print('tick_transform', tt)
            trparams = TrParams(DimParams((wt[0], wt[1]), wt[2]), DimParams((tt[0],tt[1]), tt[2]), 200)
        else:
            trparams = Trut.default_params
        
            

        # fixme: allow configuring the transforms.
        super().__init__(Rec(paths, cache=cache,
                             file_re=wash('rec_file_re'),
                             path_res=wash('rec_path_res'),
                             transpose=((wash('transpose')=='True') or False),
                             trparams=trparams,
                             det=wash('det'),
                            ),
                         Tru(paths, threshold, cache=cache,
                             file_re=wash('tru_file_re'),
                             path_res=wash('tru_path_res'),
                             transpose=((wash('transpose')=='True') or False),
                             trparams=trparams,
                             det=wash('det'),
                            )
                        )

