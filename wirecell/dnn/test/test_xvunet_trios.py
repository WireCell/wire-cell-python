#!/usr/bin/env python
'''
Tests for reading precomputed cross-plane trios into the xvunet dataset.
'''

import h5py
import numpy
import pytest
import torch

from wirecell.dnn.apps.xvunet.trios import (
    Trios, collate, batch_index, gather_all, select_by_tru)


NCHAN, NTICK = 2560, 1500


def write_trios(path, samples):
    '''
    Write {sample id: (uvwt, q)} in the layout the converter produces.
    '''
    with h5py.File(path, 'w') as fp:
        for sid, (uvwt, q) in samples.items():
            grp = fp.create_group(str(sid))
            grp.create_dataset('trio_uvwt', data=numpy.asarray(uvwt, numpy.int16))
            grp.create_dataset('trio_q', data=numpy.asarray(q, numpy.float32))
    return path


def some(k, seed=0):
    rng = numpy.random.default_rng(seed)
    uvwt = numpy.stack([rng.integers(0, 800, k),
                        rng.integers(800, 1600, k),
                        rng.integers(1600, NCHAN, k),
                        rng.integers(0, NTICK, k)], axis=1)
    return uvwt, rng.random(k)


def test_index_and_read(tmp_path):
    p = write_trios(tmp_path / 'x_7-g4-trio.h5',
                    {0: some(5, 0), 1: some(3, 1)})
    ds = Trios([str(p)])
    assert len(ds) == 2
    assert ds.sample_keys() == [('7', '0'), ('7', '1')]

    uvwt, q = ds[0]
    assert uvwt.shape == (5, 4) and uvwt.dtype is torch.int16
    assert q.shape == (5,) and q.dtype is torch.float32


def test_sorted_by_sample_id(tmp_path):
    '''
    Sample order must follow the ID, not HDF5's key order, or the trios attach
    to the wrong images.  h5py iterates alphabetically, so 10 precedes 2.
    '''
    p = write_trios(tmp_path / 'x_0-g4-trio.h5',
                    {sid: some(2, sid) for sid in (2, 10)})
    ds = Trios([str(p)])
    assert [sid for _, sid in ds.sample_keys()] == ['10', '2']


def test_missing_layer_is_an_error(tmp_path):
    path = tmp_path / 'x_0-g4-trio.h5'
    with h5py.File(path, 'w') as fp:
        fp.create_group('0').create_dataset(
            'trio_uvwt', data=numpy.zeros((3, 4), numpy.int16))
    with pytest.raises(ValueError, match='layers'):
        Trios([str(path)])


def test_mismatched_lengths_are_an_error(tmp_path):
    path = tmp_path / 'x_0-g4-trio.h5'
    with h5py.File(path, 'w') as fp:
        grp = fp.create_group('0')
        grp.create_dataset('trio_uvwt', data=numpy.zeros((3, 4), numpy.int16))
        grp.create_dataset('trio_q', data=numpy.zeros(2, numpy.float32))
    ds = Trios([str(path)])
    with pytest.raises(ValueError, match='indices vs'):
        ds[0]


def test_collate_is_flat_not_padded(tmp_path):
    '''
    The batch is concatenated with per-sample counts, so nothing is padded and
    the total length is the sum of the samples' K.
    '''
    rec = torch.zeros(1, NCHAN, NTICK)
    tru = torch.zeros(1, NCHAN, NTICK)
    batch = [(rec, tru, (torch.from_numpy(u.astype(numpy.int16)),
                         torch.from_numpy(q.astype(numpy.float32))))
             for u, q in (some(5, 0), some(3, 1), some(0, 2))]

    feats, (labels, (uvwt, q, sizes)) = collate(batch)
    assert feats.shape == (3, 1, NCHAN, NTICK)
    assert labels.shape == (3, 1, NCHAN, NTICK)
    assert uvwt.shape == (8, 4) and q.shape == (8,)
    assert sizes.tolist() == [5, 3, 0]
    assert batch_index(sizes).tolist() == [0]*5 + [1]*3


def test_gather_all_picks_the_named_pixels():
    '''
    A trio names one pixel per view; gather_all must return those values and no
    others, with the batch index implied by the counts.
    '''
    z = torch.zeros(2, 1, NCHAN, NTICK)
    uvwt = torch.tensor([[10, 900, 1700, 5],
                         [11, 901, 1701, 6],
                         [12, 902, 1702, 7]], dtype=torch.int16)
    sizes = torch.tensor([2, 1])

    # Mark each named pixel with a value encoding which trio and view it is.
    for k, row in enumerate(uvwt.tolist()):
        b = 0 if k < 2 else 1
        for view in range(3):
            z[b, 0, row[view], row[3]] = 100*k + view

    zu, zv, zw = gather_all(z, uvwt, sizes)
    assert zu.tolist() == [0., 100., 200.]
    assert zv.tolist() == [1., 101., 201.]
    assert zw.tolist() == [2., 102., 202.]


def test_gather_all_respects_the_batch_split():
    '''
    The same (row, tick) in two samples must read from its own sample.  A
    gather that dropped the batch index would pass the previous test and fail
    this one.
    '''
    z = torch.zeros(2, 1, NCHAN, NTICK)
    z[0, 0, 10, 5], z[0, 0, 900, 5], z[0, 0, 1700, 5] = 1., 2., 3.
    z[1, 0, 10, 5], z[1, 0, 900, 5], z[1, 0, 1700, 5] = -1., -2., -3.

    uvwt = torch.tensor([[10, 900, 1700, 5]]*2, dtype=torch.int16)
    zu, zv, zw = gather_all(z, uvwt, torch.tensor([1, 1]))
    assert zu.tolist() == [1., -1.]
    assert zv.tolist() == [2., -2.]
    assert zw.tolist() == [3., -3.]


def test_select_by_tru_needs_all_three():
    '''
    A trio survives only when all three of its pixels are ROI in the target.
    Each of the three single-miss cases must drop, or the correspondence weight
    would be driven by target incompleteness rather than cross-view failure.
    '''
    tru = torch.zeros(1, NCHAN, NTICK)
    rows = [[10, 900, 1700, 5],   # all three set   -> keep
            [11, 901, 1701, 6],   # U missing       -> drop
            [12, 902, 1702, 7],   # V missing       -> drop
            [13, 903, 1703, 8],   # W missing       -> drop
            [14, 904, 1704, 9]]   # none set        -> drop
    for k, row in enumerate(rows):
        for view in range(3):
            if k == 0 or (k < 4 and view != k - 1):
                tru[0, row[view], row[3]] = 1.

    uvwt = torch.tensor(rows, dtype=torch.int16)
    q = torch.arange(len(rows), dtype=torch.float32)
    kept_uvwt, kept_q = select_by_tru((uvwt, q), tru)

    assert kept_q.tolist() == [0.]
    assert kept_uvwt.tolist() == [rows[0]]
    assert kept_uvwt.dtype is torch.int16


def test_select_by_tru_keeps_everything_when_all_roi():
    tru = torch.ones(1, NCHAN, NTICK)
    uvwt, q = some(20, 3)
    uvwt = torch.from_numpy(uvwt.astype(numpy.int16))
    q = torch.from_numpy(q.astype(numpy.float32))
    kept_uvwt, kept_q = select_by_tru((uvwt, q), tru)
    assert torch.equal(kept_uvwt, uvwt) and torch.equal(kept_q, q)


def test_select_by_tru_can_empty_a_sample():
    '''
    A sample can lose every trio; collate must still handle it, since that is
    the zero-length slice case.
    '''
    tru = torch.zeros(1, NCHAN, NTICK)
    uvwt, q = some(7, 4)
    kept = select_by_tru((torch.from_numpy(uvwt.astype(numpy.int16)),
                          torch.from_numpy(q.astype(numpy.float32))), tru)
    assert kept[0].shape == (0, 4) and kept[1].shape == (0,)

    rec = torch.zeros(1, NCHAN, NTICK)
    _, (_, (cu, cq, sizes)) = collate([(rec, tru[0:1], kept)])
    assert cu.shape == (0, 4) and cq.shape == (0,) and sizes.tolist() == [0]


def test_gather_all_is_differentiable():
    '''
    The gather is the whole path from the model output to the correspondence
    term, so gradient must flow back through it to exactly the named pixels.
    '''
    z = torch.zeros(1, 1, NCHAN, NTICK, requires_grad=True)
    uvwt = torch.tensor([[10, 900, 1700, 5]], dtype=torch.int16)
    zu, zv, zw = gather_all(z, uvwt, torch.tensor([1]))
    (zu + zv + zw).sum().backward()

    assert z.grad[0, 0, 10, 5] == 1.
    assert z.grad[0, 0, 900, 5] == 1.
    assert z.grad[0, 0, 1700, 5] == 1.
    assert z.grad.abs().sum() == 3.


def test_require_tru_accepts_ini_strings(tmp_path):
    '''
    Config values arrive from an INI file as strings, and bool('false') is True,
    so a plain bool() would leave the filter on when it was asked to be off.
    '''
    from wirecell.dnn.apps.xvunet.model import _boolish
    assert _boolish('false') is False
    assert _boolish('true') is True
    assert _boolish(False) is False
