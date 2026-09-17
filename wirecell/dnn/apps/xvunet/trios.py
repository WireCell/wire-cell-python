#!/usr/bin/env python
'''
Reading precomputed cross-plane "trios" for the xvunet correspondence loss.

A trio is one channel in each plane at one tick, known to share a cause because
a single deposition produced all three.  DepoFluxSplat emits them run-length
encoded; a converter puts them on the training grid as HDF5 holding, per sample:

    /{sample}/trio_uvwt   (K,4) int16    row_U, row_V, row_W, tick
    /{sample}/trio_q      (K,)  float32  charge

The rows already index the concatenated (U,V,W) channel axis the dataset
builds, so consuming a trio is three gathers into the model's (B,1,2560,1500)
tensor and no geometry is needed here.

WHY THIS IS NOT AN hdf.Single.  Single stacks a sample's layers with
torch.stack and transposes to (nlayer, nchan, ntick), which assumes every layer
is the same frame-shaped array.  trio_uvwt and trio_q are neither the same
shape as each other nor fixed across samples, so they need their own reader.

RAGGED HANDLING.  K varies per sample -- 0.9e6 to 3.0e6 on the sample measured
-- so default_collate cannot stack it.  Rather than padding every sample to
K_max and carrying a validity mask, `collate` concatenates the batch into one
flat list and records each sample's length.  At K this size padding to the
batch maximum would waste a large fraction of the transfer for nothing, and the
flat form also makes the loss reductions plain means over a 1-D tensor instead
of masked means over a padded one.  The per-sample batch index is rebuilt on
device with repeat_interleave, which is far cheaper than shipping it.
'''

from collections import OrderedDict, defaultdict

import h5py
import torch
from torch.utils.data import Dataset as TorchDataset

from wirecell.dnn.data.hdf import ReMatcher, allkeys

import logging
log = logging.getLogger("wirecell.dnn")


class Trios(TorchDataset):
    '''
    A dataset yielding one sample's (uvwt, q) trio tensors.

    uvwt is (K,4) int16 and q is (K,) float32, both straight from file: no
    transform, no stacking, no transpose.
    '''

    # Matches hdf.Single: reopening is cheap next to reading a sample, and
    # holding every file open pins HDF5's per-file caches.
    max_open = 4

    default_file_re = r'.*_(\d+)-g4-trio\.h5'
    default_path_res = (r'/(\d+)/trio_uvwt', r'/(\d+)/trio_q')

    def __init__(self, paths, file_re=None, path_res=None, cache=False):
        self.match = ReMatcher(file_re or self.default_file_re,
                               path_res or self.default_path_res)
        self._index = list()    # idx -> [(fname, fkey), ...] in layer order
        self._cache = dict()
        self._open = OrderedDict()
        self._do_cache = cache
        if paths:
            self.append(paths)
        log.debug(f'xvunet trios: {len(self)} samples')

    def _file(self, fname):
        fp = self._open.get(fname, None)
        if fp is not None:
            self._open.move_to_end(fname)
            return fp
        fp = self._open[fname] = h5py.File(fname, 'r')
        while len(self._open) > self.max_open:
            _, old = self._open.popitem(last=False)
            old.close()
        return fp

    def __getstate__(self):
        # Open HDF5 handles neither pickle nor survive a fork, so DataLoader
        # workers must reopen on demand.
        state = dict(self.__dict__)
        state['_open'] = OrderedDict()
        return state

    def append(self, paths):
        byids = defaultdict(dict)   # (fid,sid) -> lid -> (fname,fkey)
        for fname in paths:
            with h5py.File(fname, 'r') as fp:
                for fkey in allkeys(fp):
                    if not isinstance(fp.get(fkey), h5py.Dataset):
                        continue
                    got = self.match(fname, fkey)
                    if not got:
                        continue
                    fid, sid, lid = got
                    byids[(fid, sid)][lid] = (fname, fkey)

        for kid in sorted(byids):
            entry = byids[kid]
            if len(entry) != len(self.match):
                raise ValueError(
                    f'trio sample {kid} has {len(entry)} of '
                    f'{len(self.match)} layers: {sorted(entry.values())}')
            self._index.append([entry[lid] for lid in sorted(entry)])

    def sample_keys(self):
        '''
        The ordered (file ID, sample ID) keys, for alignment checks.
        '''
        keys = list()
        for layers in self._index:
            fname, fkey = layers[0]
            fid, sid, _ = self.match(fname, fkey)
            keys.append((fid, sid))
        return keys

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        got = self._cache.get(idx, None)
        if got is not None:
            return got

        (uf, uk), (qf, qk) = self._index[idx]
        uvwt = torch.from_numpy(self._file(uf)[uk][:])
        q = torch.from_numpy(self._file(qf)[qk][:]).reshape(-1)
        if uvwt.shape[0] != q.shape[0]:
            raise ValueError(f'trio sample {idx}: {uvwt.shape[0]} indices vs '
                             f'{q.shape[0]} charges in {uf}')

        got = (uvwt, q)
        if self._do_cache:
            self._cache[idx] = got
        return got


def select_by_tru(trio, tru):
    '''
    Drop trios that are not ROI in all three views.

    The correspondence term reads a trio as "these three pixels share a cause,
    so a plane that missed while its partners hit is the interesting failure".
    That reading needs all three partners to be pixels the target says should
    fire; where a partner's target is 0 a confident 0 is the correct answer,
    not evidence that anything was missed.

    trio is (uvwt, q) for one sample and tru is that sample's (1, nchan, ntick)
    target, already thresholded to {0,1} by the Tru transform, so this is three
    gathers and an and.
    '''
    uvwt, q = trio
    idx = uvwt.long()
    tick = idx[:, 3]
    keep = torch.ones(idx.shape[0], dtype=torch.bool)
    for view in range(3):
        keep &= tru[0, idx[:, view], tick] > 0
    return uvwt[keep], q[keep]


def collate(batch):
    '''
    Collate samples of (rec, tru, (uvwt, q)) into batched tensors.

    rec and tru stack as usual.  The trios of the batch are concatenated into
    one flat list plus the per-sample counts, since K varies per sample:

        uvwt   (K_total, 4) int16
        q      (K_total,)   float32
        sizes  (B,)         int64

    Rebuild the batch index on device with

        b = torch.repeat_interleave(torch.arange(len(sizes)), sizes)

    which is cheaper than transferring it.  A sample with no trios contributes
    a zero-length slice and simply drops out of the reduction.
    '''
    recs, trus, trios = zip(*batch)
    rec = torch.stack(recs)
    tru = None if trus[0] is None else torch.stack(trus)

    uvwt = torch.cat([t[0] for t in trios])
    q = torch.cat([t[1] for t in trios])
    sizes = torch.tensor([t[0].shape[0] for t in trios], dtype=torch.int64)
    return rec, (tru, (uvwt, q, sizes))


def batch_index(sizes, device=None):
    '''
    The per-trio sample index implied by the collated per-sample counts.
    '''
    return torch.repeat_interleave(
        torch.arange(len(sizes), device=device or sizes.device),
        sizes.to(device or sizes.device))


def gather_all(z, uvwt, sizes):
    '''
    Gather the three per-view logits for every trio in a collated batch.

    z is the model output, (B, 1, nchan, ntick).  Returns (zu, zv, zw), each
    (K_total,), in logit space: this deliberately does not materialise a
    full-image per-pixel loss tensor, which would be 15 MB per sample at fp32
    for values that are then almost entirely discarded.
    '''
    b = batch_index(sizes, z.device)
    idx = uvwt.to(z.device, non_blocking=True).long()
    tick = idx[:, 3]
    return tuple(z[b, 0, idx[:, view], tick] for view in range(3))
