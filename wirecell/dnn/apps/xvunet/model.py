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

    #: cfg keys that only seed a fresh model.  A full checkpoint overwrites
    #: whatever they produced, so honouring them on resume is wasted I/O (three
    #: trunk files, ~40M params, per rank under DDP) and misleading: --load wins
    #: silently, so a user who thinks their unet_checkpoints took effect is wrong.
    INIT_KEYS = ('unet_checkpoints', 'init_checkpoint')

    #: cfg keys fixing parameter shapes or the training regime.  A resume must
    #: agree with the checkpoint on all of these or the restored optimizer state
    #: is invalid.  freeze_unets is here because it decides which parameters the
    #: optimizer is even built over (52 tensors frozen vs 274 unfrozen on a small
    #: model), so a mismatch otherwise surfaces far away as an opaque param-group
    #: error.  attn_mode, use_checkpoint and checkpoint_trunks are deliberately
    #: absent: they change neither parameter shapes nor the parameter set.
    STRUCTURAL_KEYS = ('view_splits', 'chunks', 'd_model', 'n_heads', 'n_layers',
                       'band', 'ffn_mult', 'n_input_channels', 'n_classes',
                       'freeze_unets')

    @classmethod
    def _kwds(cls, cfg):
        '''
        Coerce a config section to XViewUNet keyword arguments, applying the
        defaults.  Values may arrive as INI strings or as native types, so this
        is also what makes "32" and 32, or "true" and True, compare equal.
        '''
        return dict(
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

    @classmethod
    def resolve_config(cls, cfg, checkpoint_args=None):
        '''
        Reconcile a config section with the checkpoint a run is resuming from,
        returning the config to actually build with.

        Called only when resuming; a fresh run gets its config unchanged.  Two
        things happen here, both of which need to know what the keys mean and so
        do not belong in the generic harness:

        - INIT_KEYS are dropped, since --load overwrites what they would seed.
        - STRUCTURAL_KEYS are checked against what the checkpoint recorded,
          failing here with a message naming the offending key rather than later
          as an opaque optimizer param-group error.

        checkpoint_args is the model config the checkpoint recorded, or None if
        it recorded none -- an old checkpoint, in which case the resume proceeds
        unvalidated rather than being stranded.

        Changing regime is not a resume: -l/--load continues one run, optimizer
        moments and all, so it requires the same regime.  To go from a frozen
        stage-1 to an unfrozen stage-2, name the stage-1 file as the [model]
        init_checkpoint instead -- that takes the weights with a fresh optimizer
        built over the newly trainable parameters.

        Note both sides are normalised with today's defaults, so a key absent
        from both compares equal even if its default has changed since the
        checkpoint was written.  Keys are only added to STRUCTURAL_KEYS, never
        removed, so the failure mode is a missed mismatch, not a false one.
        '''
        cfg = dict(cfg)

        dropped = [k for k in cls.INIT_KEYS if cfg.pop(k, None) is not None]
        if dropped:
            log.info(f'xvunet: resuming, so ignoring {", ".join(dropped)} -- '
                     'the loaded checkpoint supersedes what they would seed')

        if checkpoint_args is None:
            log.warning('xvunet: this checkpoint records no model config, so '
                        'the resume cannot be validated against it.  Any '
                        'mismatch will surface later as an optimizer '
                        'param-group error instead.')
            return cfg

        want, got = cls._kwds(checkpoint_args), cls._kwds(cfg)
        bad = [k for k in cls.STRUCTURAL_KEYS if want[k] != got[k]]
        if not bad:
            return cfg

        detail = '; '.join(f'{k}: checkpoint={want[k]!r} config={got[k]!r}'
                           for k in bad)
        msg = ('xvunet: cannot resume from this checkpoint, it disagrees with '
               f'the config on {detail}')
        if 'freeze_unets' in bad:
            msg += ('.  freeze_unets decides which parameters the optimizer is '
                    'built over, so its state cannot carry across the change.  '
                    'To move between stages use init_checkpoint, which takes '
                    'the weights with a fresh optimizer; -l/--load is for '
                    'resuming within one regime')
        raise ValueError(msg)

    def __init__(self, **cfg):
        super().__init__()
        # cfg = model_config or dict()

        kwds = self._kwds(cfg)
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
