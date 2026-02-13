#!/usr/bin/env python
from torch import optim, tensor, float32 as tf32

## The "app" API
from wirecell.dnn.models.unet_crossview import UNetCrossView
from .data import Dataset

DO_LOOPER=False
MP_OUT=False
# DO_LOOPER = DO_LOOPER or MP_OUT #Not enough memory if doing mp out
if DO_LOOPER:
    from wirecell.dnn.trainers.train import Looper as Trainer
else:
    from wirecell.dnn.trainers.train import Classifier as Trainer

from torch.nn import BCELoss as Criterion

def Optimizer(params):
    return optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=0.0005)

def Network():
    return UNetCrossView(
 
        wires_file='protodunehd-wires-larsoft-v1.json.bz2',
        chanmap_file=2560,
        nchans=[800, 800, 480, 480],
        det_type='hd',
        cells_file='pdhd_cells_fixed.pt',
        mp_out=MP_OUT,
        scatter_out=False,
        output_as_tuple=False,

        n_unet_features=4,
        checkpoint=False,
        n_feat_wire = 0,
        detector=0,
        n_input_features=1,

        network_style='U-MP-U',
        # network_style='U-U',
        # network_style='U',
    )

class Criterion:
    '''Multi-term loss'''
    def __init__(self):
        from torch.nn import BCELoss
        self.crit = BCELoss(reduction='sum') if DO_LOOPER else BCELoss()
    def to(self, device):
        self.crit = self.crit.to(device)
        return self

    def __call__(self, prediction : tuple, label : tuple, do_norm : bool = False ):
        if len(prediction) != len(label):
            raise Exception(
            'Error! Expected prediction and label to have same size'
            f'but received {len(prediction)} and {len(label)} respectively.')
        loss = 0
        for i, (p, l) in enumerate(zip(prediction, label)):
            norm = 1./(l.shape[-2]) if do_norm else 1.
            loss += self.crit(p, l)*norm
        return loss