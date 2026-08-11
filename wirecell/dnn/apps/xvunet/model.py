#!/usr/bin/env python
'''
Network wrapper adapting XViewUNet to the app API and INI-string config.
'''
import ast

import torch.nn as nn

from wirecell.dnn.models.xvunet import XViewUNet

import logging
log = logging.getLogger("wirecell.dnn")


def _wash(config, key, default=None):
    '''
    Get a config value, evaluating INI string values that hold python
    list/tuple/dict literals.
    '''
    val = config.get(key, default)
    if isinstance(val, str) and val.lstrip().startswith(('[', '(', '{')):
        val = ast.literal_eval(val)
    return val


def _boolish(val):
    '''
    Interpret an INI value as a bool; "1"/"true"/"yes"/"on" are true.
    '''
    if isinstance(val, str):
        return val.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(val)


class Network(nn.Module):
    '''
    The app-API model: an XViewUNet built from an INI-string config.

    Keys and defaults mirror XViewUNet's signature.  Values arrive as strings,
    so list/tuple/dict literals go through _wash and booleans through _boolish.

    The model is held as self.xvunet, so a checkpoint saved from this wrapper
    carries an "xvunet." key prefix -- which is what
    XViewUNet.load_full_checkpoint strips when resuming from one.
    '''

    def __init__(self, model_config=None):
        super().__init__()
        cfg = model_config or dict()

        kwds = dict(
            view_splits=_wash(cfg, 'view_splits', [[800], [800], [480, 480]]),
            chunks=_wash(cfg, 'chunks', [8, 8, 8]),
            d_model=int(cfg.get('d_model', 96)),
            n_heads=int(cfg.get('n_heads', 4)),
            n_layers=int(cfg.get('n_layers', 2)),
            band=int(cfg.get('band', 1)),
            ffn_mult=int(cfg.get('ffn_mult', 4)),
            n_input_channels=int(cfg.get('n_input_channels', 1)),
            n_classes=int(cfg.get('n_classes', 1)),
            unet_checkpoints=_wash(cfg, 'unet_checkpoints'),
            freeze_unets=_boolish(cfg.get('freeze_unets', False)),
            init_checkpoint=cfg.get('init_checkpoint'),
            use_checkpoint=_boolish(cfg.get('use_checkpoint', True)),
            checkpoint_trunks=_boolish(cfg.get('checkpoint_trunks', False)),
        )
        log.info(f'xvunet network: {kwds}')
        self.xvunet = XViewUNet(**kwds)

        # Attention scope is runtime state, not a constructor argument, so a
        # checkpoint stays loadable under any mode.  Default 'all' forbids
        # attention between two faces of one view; set attn_mode=legacy to
        # reproduce a model trained before modes were added.
        self.set_attention_mode(cfg.get('attn_mode', 'all'))

    def set_attention_mode(self, mode):
        '''Select the attention scope; see xvunet.ATTN_MODES.'''
        self.xvunet.set_attention_mode(mode)
        return self

    def forward(self, x):
        return self.xvunet(x)
