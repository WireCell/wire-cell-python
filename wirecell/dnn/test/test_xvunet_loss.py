#!/usr/bin/env python
'''
Tests for the xvunet criterion: BCE plus the cross-plane correspondence term.
'''

import pytest
import torch
import torch.nn as nn

from wirecell.dnn.apps.xvunet.loss import Criterion


NCHAN, NTICK = 64, 32          # small: none of this depends on the real size


def empty_trios():
    return (torch.zeros(0, 4, dtype=torch.int16),
            torch.zeros(0), torch.tensor([1]))


def one_trio(rows=(10, 20, 30), tick=5):
    uvwt = torch.tensor([list(rows) + [tick]], dtype=torch.int16)
    return uvwt, torch.ones(1), torch.tensor([1])


def scene(zu, zv, zw, rows=(10, 20, 30), tick=5, targets=(1., 1., 1.)):
    '''
    A one-sample batch with a single trio at known logits and targets.
    '''
    logits = torch.zeros(1, 1, NCHAN, NTICK, requires_grad=True)
    tru = torch.zeros(1, 1, NCHAN, NTICK)
    with torch.no_grad():
        for row, z in zip(rows, (zu, zv, zw)):
            logits[0, 0, row, tick] = z
    for row, t in zip(rows, targets):
        tru[0, 0, row, tick] = t
    return logits, tru, one_trio(rows, tick)


def test_tensor_labels_are_plain_bce():
    '''
    Without trios the criterion must be exactly BCEWithLogitsLoss, so every
    other app and any xvunet run with trios unconfigured is unchanged.
    '''
    crit = Criterion()
    logits = torch.randn(2, 1, NCHAN, NTICK)
    tru = (torch.rand(2, 1, NCHAN, NTICK) > 0.7).float()
    assert torch.equal(crit(logits, tru), nn.BCEWithLogitsLoss()(logits, tru))


def test_zero_strength_matches_base():
    crit = Criterion(trio_strength=0.0)
    logits, tru, trios = scene(-4., 4., 4.)
    assert torch.equal(crit(logits, (tru, trios)),
                       nn.BCEWithLogitsLoss()(logits, tru))


def test_empty_trios_matches_base():
    '''
    A sample can lose every trio to the ROI filter; that must not produce a nan
    from a mean over nothing.
    '''
    crit = Criterion(trio_strength=1.0)
    logits = torch.randn(1, 1, NCHAN, NTICK)
    tru = torch.zeros(1, 1, NCHAN, NTICK)
    out = crit(logits, (tru, empty_trios()))
    assert torch.isfinite(out)
    assert torch.equal(out, nn.BCEWithLogitsLoss()(logits, tru))


def test_one_missed_outweighs_all_missed():
    '''
    The whole point of the term.  A plane that missed while its partners hit
    must be penalised far more than the same miss when all three missed, which
    is a hard hit rather than a cross-view failure.
    '''
    crit = Criterion(trio_strength=1.0)

    one = crit.trio_term(*scene(-4., 4., 4.)[:2], scene(-4., 4., 4.)[2])
    all3 = crit.trio_term(*scene(-4., -4., -4.)[:2], scene(-4., -4., -4.)[2])
    assert one > 10 * all3


def test_all_correct_is_small():
    crit = Criterion(trio_strength=1.0)
    logits, tru, trios = scene(6., 6., 6.)
    assert crit.trio_term(logits, tru, trios) < 0.01


def _undetached_term(logits, tru, trios):
    '''
    The same term with the weight left differentiable, as a reference for what
    detaching buys.  Deliberately not importable from loss.py -- it exists only
    to be shown wrong.
    '''
    from wirecell.dnn.apps.xvunet.trios import gather_all
    import torch.nn.functional as F
    zs = gather_all(logits, trios[0], trios[2])
    ts = gather_all(tru, trios[0], trios[2])
    succ = [torch.where(t > 0, torch.sigmoid(z), 1.0 - torch.sigmoid(z))
            for t, z in zip(ts, zs)]
    total = logits.new_zeros(())
    for view in range(3):
        weight = succ[(view + 1) % 3] * succ[(view + 2) % 3]
        bce = F.binary_cross_entropy_with_logits(zs[view], ts[view],
                                                 reduction='none')
        total = total + (weight * bce).mean()
    return total / 3.0


def test_weight_is_detached_so_partners_are_not_pushed_down():
    '''
    The failure this guards against: if the weight were differentiable, a
    confident partner would raise w for its neighbour's large BCE, and the
    cheapest way to cut the loss would be to LOWER the partner -- breaking a
    right answer to reach agreement.

    That is not a hypothetical.  With u missed and v, w confident, the
    undetached gradient on z_v comes out POSITIVE (descent lowers a correct
    plane) while the detached one is negative (descent raises it further).  All
    three targets are 1 here, so every correctly-signed gradient is negative.
    '''
    crit = Criterion(trio_strength=1.0)

    logits, tru, trios = scene(-6., 5., 5.)
    crit.trio_term(logits, tru, trios).backward()
    detached = logits.grad.clone()

    logits2, tru2, trios2 = scene(-6., 5., 5.)
    _undetached_term(logits2, tru2, trios2).backward()
    undetached = logits2.grad

    assert undetached[0, 0, 20, 5] > 0       # v would be pushed DOWN
    assert detached[0, 0, 20, 5] < 0         # v is pushed further up
    assert detached[0, 0, 30, 5] < 0         # w likewise
    assert detached[0, 0, 10, 5] < 0         # u, the miss, also raised


def test_missed_pixel_gets_the_larger_gradient():
    '''
    Gradient must concentrate on the plane that missed, not on the two that
    were already right.
    '''
    crit = Criterion(trio_strength=1.0)
    logits, tru, trios = scene(-6., 5., 5.)
    crit.trio_term(logits, tru, trios).backward()

    gu = logits.grad[0, 0, 10, 5].abs()
    gv = logits.grad[0, 0, 20, 5].abs()
    assert gu > 100 * gv


def test_term_is_scale_free_in_trio_count():
    '''
    The term is a mean, so duplicating every trio must not change it.  A sum
    would make the loss depend on how many deposits an event happened to have.
    '''
    crit = Criterion(trio_strength=1.0)
    logits, tru, _ = scene(-4., 4., 4.)
    rows, tick = (10, 20, 30), 5

    one = (torch.tensor([list(rows) + [tick]], dtype=torch.int16),
           torch.ones(1), torch.tensor([1]))
    two = (torch.tensor([list(rows) + [tick]] * 2, dtype=torch.int16),
           torch.ones(2), torch.tensor([2]))
    assert torch.allclose(crit.trio_term(logits, tru, one),
                          crit.trio_term(logits, tru, two))


def test_zero_target_partner_counts_as_success_when_predicted_zero():
    '''
    With the dataset filter off, a partner whose target is 0 and whose
    prediction is 0 did WELL, and should carry weight like any other correct
    partner.  Reading its raw probability instead would call it a miss.
    '''
    crit = Criterion(trio_strength=1.0)
    # v has target 0 and predicts 0 (correct); w is correct; u missed.
    right = crit.trio_term(*scene(-6., -6., 5., targets=(1., 0., 1.))[:2],
                           scene(-6., -6., 5., targets=(1., 0., 1.))[2])
    # Same but v wrongly fires: it is now a bad partner, so u's weight drops.
    wrong = crit.trio_term(*scene(-6., 6., 5., targets=(1., 0., 1.))[:2],
                           scene(-6., 6., 5., targets=(1., 0., 1.))[2])
    assert right > wrong
