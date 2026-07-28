#!/usr/bin/env python
'''
Three-view dataset for xvunet.

Each sample concatenates the per-view "rec" (deconvolved) images along the
electronics-channel axis into a single (1, sum-of-channels, nticks) tensor,
and likewise the "tru" (target) images.  The per-view images are found in
separate per-view HDF5 "frame files" (e.g. ...-g4-rec-0.h5 / -1 / -2 for
U/V/W) which must be strictly parallel: same file IDs and same sample IDs in
every view.
'''

import torch
from torch.utils.data import Dataset as TorchDataset

from wirecell.dnn.data import hdf

from .transforms import Rec as Rect, Tru as Trut, Params as TrParams, DimParams

import logging
log = logging.getLogger("wirecell.dnn")


class Rec(hdf.Single):
    '''
    One view's "rec" dataset: deconvolved (loose LF) images.
    '''

    path_res = (r'/(\d+)/frame_loose_lf\d',)

    def __init__(self, paths, file_re,
                 path_res=None,
                 trparams: TrParams = None, cache=False):
        dom = hdf.Domain(hdf.ReMatcher(file_re, path_res or self.path_res),
                         transform=Rect(trparams, transpose=True),
                         cache=cache, grad=False,
                         name="xvunetrec")
        super().__init__(dom, paths)


class Tru(hdf.Single):
    '''
    One view's "tru" dataset: the target ROI.
    '''

    path_res = (r'/(\d+)/frame_ductor\d',)

    def __init__(self, paths, file_re, threshold=0.5,
                 path_res=None,
                 trparams: TrParams = None, cache=False):
        dom = hdf.Domain(hdf.ReMatcher(file_re, path_res or self.path_res),
                         transform=Trut(trparams, True, threshold),
                         cache=cache, grad=False,
                         name="xvunettru")
        super().__init__(dom, paths)


def _sample_keys(single):
    '''
    The ordered (file ID, sample ID) keys of an hdf.Single.
    '''
    keys = list()
    for layers in single._index:
        fname, fkey = layers[0]
        fid, sid, lid = single.domain.match(fname, fkey)
        keys.append((fid, sid))
    return keys


class Dataset(TorchDataset):
    '''
    zip over views and over (rec, tru), concatenating views channel-wise.
    '''

    default_rec_file_res = tuple(
        r'.*_(\d+)-g4-rec-%d\.h5' % view for view in range(3))
    default_tru_file_res = tuple(
        r'.*_(\d+)-g4-tru-%d\.h5' % view for view in range(3))
    default_elech_binnings = ((0, 800, 1), (0, 800, 1), (0, 960, 1))
    default_tick_binning = (0, 1500, 1)

    def __init__(self, paths, threshold=0.5, cache=False, config=None):
        config = config or dict()

        def wash(key, default=None):
            val = config.get(key, default)
            if isinstance(val, str) and val.lstrip().startswith(('[', '(', '{')):
                val = eval(val)  # same idiom as the other apps
                log.debug(f'xvunet dataset {key} = {val}')
            return val

        rec_file_res = wash('rec_file_res', self.default_rec_file_res)
        tru_file_res = wash('tru_file_res', self.default_tru_file_res)
        rec_path_res = wash('rec_path_res')
        tru_path_res = wash('tru_path_res')
        elech_binnings = wash('elech_binnings', self.default_elech_binnings)
        tick_binning = wash('tick_binning', self.default_tick_binning)
        rec_norm = float(config.get('rec_norm', 1.0))
        tru_norm = float(config.get('tru_norm', 0.05))
        threshold = float(config.get('threshold', threshold))

        nviews = len(rec_file_res)
        if not (len(tru_file_res) == len(elech_binnings) == nviews):
            raise ValueError(
                f'inconsistent per-view config: {nviews} rec_file_res, '
                f'{len(tru_file_res)} tru_file_res, '
                f'{len(elech_binnings)} elech_binnings')

        tick = DimParams([int(i) for i in tick_binning[0:2]], int(tick_binning[2]))

        self._recs = list()
        self._trus = list()
        for view in range(nviews):
            eb = [int(i) for i in elech_binnings[view]]
            elech = DimParams(eb[0:2], eb[2])
            self._recs.append(Rec(paths, rec_file_res[view],
                                  path_res=rec_path_res,
                                  trparams=TrParams(elech, tick, rec_norm),
                                  cache=cache))
            self._trus.append(Tru(paths, tru_file_res[view],
                                  threshold=threshold,
                                  path_res=tru_path_res,
                                  trparams=TrParams(elech, tick, tru_norm),
                                  cache=cache))

        self._check_alignment()
        log.info(f'xvunet dataset: {nviews} views, {len(self)} samples')

    def _check_alignment(self):
        '''
        All views' rec and tru Singles must index the same (file, sample) IDs
        in the same order.
        '''
        singles = self._recs + self._trus
        ref = _sample_keys(singles[0])
        if not ref:
            raise ValueError('xvunet dataset is empty: check files and regexes')
        for single in singles[1:]:
            got = _sample_keys(single)
            if got != ref:
                name = single.domain.name
                fr = single.domain.match.file_re.pattern
                raise ValueError(
                    f'misaligned samples in {name} ({fr}): '
                    f'{len(got)} samples vs {len(ref)} expected; '
                    f'first difference: '
                    f'{next((a, b) for a, b in zip(got, ref) if a != b) if got else None}')

    def __len__(self):
        return len(self._recs[0])

    def __getitem__(self, idx):
        rec = torch.cat([one[idx] for one in self._recs], dim=1)
        tru = torch.cat([one[idx] for one in self._trus], dim=1)
        return rec, tru
