#!/usr/bin/env python
'''
The xvunet criterion: per-pixel BCE plus a cross-plane correspondence term.

A blob of charge is seen by all three planes at once -- one U wire, one V wire,
one W wire, same tick.  Per-pixel BCE grades pixels independently and knows
nothing of that, so missing the W pixel of a trio counts the same whether its U
and V partners were found or not.  That is exactly the failure cross-view
attention exists to fix, so it is where the gradient should be pointed.

THIS IS EMPHASIS, NOT NEW SUPERVISION.  The trios are a function of the truth,
so BCE could in principle learn all of it.  The term redirects gradient onto
cross-view-resolvable errors, and should be judged on that.

THE SHAPE OF THE TERM.  Ordinary BCE is kept, and a trio pixel's BCE is
additionally weighted by how well its two partners did:

    w_u = s_v * s_w      s_x = p_x if the target is 1, else 1 - p_x

with the partner probabilities DETACHED.  Two planes found it and the third
missed -> the miss carries weight near 1 and counts heavily.  All three missed
-> every weight is near 0 and the trio is graded by the base BCE alone, which
is right: that is a hard hit, not a failure to carry information across planes.

Because s is the partner's success rather than its raw confidence, the term
stays meaningful if trios whose targets are not all ROI are ever let through
(dataset trio_require_tru=False).  With the default filter every target is 1
and s reduces to p.

WHAT WAS REJECTED.  A symmetric "penalise the three planes for disagreeing"
term, e.g. |pu-pv| + |pv-pw| + |pu-pw|.  It is wrong in three ways: it equals
2*(max-min) so it cannot see the middle value and scores one-plane-missed the
same as two; its gradient pushes the CORRECT planes down as hard as it lifts
the missed one, so it will break two right answers to reach agreement; and it
is minimised by ANY constant output, so predicting nothing everywhere satisfies
it perfectly.  The reweighting has none of these -- it is still BCE underneath,
minimised by being correct rather than by being uniform, and the weight is
detached so it cannot be gamed by suppressing a partner.

IS THE POPULATION THERE?  Measured with the warm-started trunks on the
regenerated data, over 2.6e6 trios: 88.88% already have all three planes
predicting ROI, 9.63% have exactly two right and one missed, and 1.49% have one
or none.  So 87% of all trio errors are cross-view-resolvable -- when this model
misses a trio pixel it usually had both partners right -- which is the error
mode this term exists to attack, and it is the dominant one.

COST.  Working in logit space and gathering is deliberate.  Running
BCEWithLogitsLoss(reduction='none') over the whole image and then weighting
would materialise a (B,1,2560,1500) loss tensor, 15 MB per sample at fp32, for
values almost all of which are discarded.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from .trios import gather_all

import logging
log = logging.getLogger("wirecell.dnn")


class Criterion(nn.Module):
    '''
    BCEWithLogits over the image, plus the trio correspondence term.

    With labels that are a plain tensor -- any app without trios, or xvunet
    with the dataset's trio_file_re unset -- this is exactly
    BCEWithLogitsLoss and nothing else runs.  The extra term appears only when
    the dataset supplies trios, and trio_strength=0 turns it off while leaving
    the rest untouched.
    '''

    # MEASURED with the real warm-started trunks on the regenerated data, batch
    # 1, 6 samples: base BCE median 0.00772 against an unweighted trio term of
    # 0.09297, so the raw term runs ~11.5x the base.  An earlier estimate from
    # truth-derived logits put that ratio near 1 and was misleading, because a
    # uniform margin on every pixel hides the two things that matter here: the
    # real model's background is easy, which makes base BCE small, and its trio
    # misses have confident partners, which makes the term large.  0.08 puts the
    # two at roughly parity at the start of stage 1.  A STARTING POINT, not a
    # tuned value -- the A/B against trio_strength=0 is what settles it.
    default_strength = 0.08

    def __init__(self, trio_strength=default_strength, pos_weight=None):
        super().__init__()
        self.trio_strength = float(trio_strength)
        pw = None if pos_weight is None else torch.tensor(float(pos_weight))
        self.base = nn.BCEWithLogitsLoss(pos_weight=pw)
        if self.trio_strength:
            log.info(f'xvunet criterion: trio_strength={self.trio_strength}')

    def trio_term(self, logits, tru, trios):
        '''
        The correspondence term alone, so it can be logged beside the base.

        Returns a scalar; zero when there are no trios in the batch.
        '''
        uvwt, _q, sizes = trios
        if uvwt.numel() == 0:
            return logits.new_zeros(())

        zs = gather_all(logits, uvwt, sizes)
        # The target at each trio pixel.  Under the default dataset filter
        # these are all 1, but gathering them keeps the term correct if the
        # filter is turned off rather than silently assuming a positive.
        ts = gather_all(tru, uvwt, sizes)

        # How well each plane did, detached: the weight grades the partners'
        # performance, it is not a quantity to optimise through.  Doing so
        # would let the model lower a partner to cheapen its own miss.
        succ = [torch.where(t > 0, p, 1.0 - p)
                for t, p in ((t, torch.sigmoid(z).detach())
                             for t, z in zip(ts, zs))]

        total = logits.new_zeros(())
        for view in range(3):
            weight = succ[(view + 1) % 3] * succ[(view + 2) % 3]
            bce = F.binary_cross_entropy_with_logits(
                zs[view], ts[view], reduction='none')
            total = total + (weight * bce).mean()
        return total / 3.0

    def forward(self, logits, labels):
        if torch.is_tensor(labels):
            return self.base(logits, labels)

        tru, trios = labels
        loss = self.base(logits, tru)
        if self.trio_strength:
            loss = loss + self.trio_strength * self.trio_term(logits, tru, trios)
        return loss
