#!/usr/bin/env python
'''
The Ronneberger, Fischer and Brox U-Net by default.

https://arxiv.org/abs/1505.04597
https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png

The following labels are used to identify units of the network and refers to the
u-net-architecture.png figure.

- dconv :: "double convolution" (dark blue arrow pair), two sub units of 3x3
  convolution + ReLU.  This makes up each major unit of the "U".  The output of
  this fans out to "dsamp" and to "skip".

- dsamp :: "down sampling" (red arrow), the 2x2 max pool downsampling that
  connects output of one dconv to input of next dconv on the downward leg of the
  "U".

- bottom :: the dconv making up the apex or bottom of the U.

- usamp :: "up sampling" (green arrow), the "up-conv 2x2" that input from dconv
  result and output to umerge.

- skip :: "skip connection" (gray arrow), this simply shunts the output from a
  dconv on the downward leg to one input of a umerge.

- umerge :: "up merge" (gray+green arrows), concatenation of the skip result
  with the up samping result and provides input to an dconv on the upward leg.

The default configuration produces U-Net.  The following optional extensions,
off by default, are supported:

- batch_norm=True :: insert two BatchNorm2d in double convolution unit (dconv).
- bilinear=True :: use bilinear interpolation instead of ConvTranspose2d in up-conv
- padding=True :: zero-pad in dconv so image input size is retained and in umerge is needed to match arrays from skip and below connections.
- nskips=N :: use a different number of skip connection levels besides 4.
- use non-square images.

'''


from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn.functional import pad as nnpad

import logging
log = logging.getLogger("wirecell.dnn")


def dconv(in_channels, out_channels, kernel_size = 3, padding = 0,
          batch_norm=False):
    '''
    The double "conv 3x3, ReLU" unit.
    '''
    parts = [
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
        nn.ReLU(inplace=True)
    ]

    if batch_norm:
        parts.insert(3, nn.BatchNorm2d(out_channels))
        parts.insert(1, nn.BatchNorm2d(out_channels))

    return nn.Sequential(*parts)


def dsamp():
    '''
    The "down sampling".
    '''
    return nn.MaxPool2d(2)


class umerge(nn.Module):
    '''
    The "upsample merge" of the outputs from a skip and a dconv.

    The "up" array is upsampled and then appended to the "over" array.

    Both options have large repercussion on upstream nodes:

    If bilinear, the number of channels in the upsampled array is unchanged else
    it is halved.  

    If padded, the upsampled array pixel dimensions will be padded to match
    those of the "over" array.
    '''
    def __init__(self, nchannels, bilinear=False, padding=False):
        '''
        Give number of channels in the input to the upsampling port.
        '''
        super().__init__()
        self.padding = padding
        if bilinear:
            self.upsamp = nn.Upsample(scale_factor=2., mode='bilinear', align_corners=True)
        else:
            self.upsamp = nn.ConvTranspose2d(nchannels, nchannels//2, 2, stride=2)

    def forward(self, over: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        up = self.upsamp(up)

        # These amounts were once computed on the first forward and cached on
        # self.  That silently reused the first input's geometry for every later
        # input of a different size, and it made the module untraceable (the
        # cache-filling branch runs only once, so the graph differed between
        # trace and its check re-run).  Recomputing costs two subtractions.
        if self.padding:
            # when not cropping we must pad special to match when target is odd size
            dw = over.size(3) - up.size(3)
            dh = over.size(2) - up.size(2)
            # nnpad takes the last dim first, matching the old [-1, -2] order.
            pads: List[int] = [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2]
            up = nnpad(up, pads)
        else:
            # centre-crop "over" down to "up"'s spatial size
            h = up.size(2)
            w = up.size(3)
            over = over.narrow(2, (over.size(2) - h) // 2, h) \
                       .narrow(3, (over.size(3) - w) // 2, w)

        cat = torch.cat((over, up), dim=1)
        return cat


class _DownBlock(nn.Module):
    '''
    One rung of the downward leg: the dconv whose output feeds a skip, plus the
    dsamp feeding the rung below.

    This exists so the legs can live in nn.ModuleLists.  TorchScript can iterate
    a ModuleList (it unrolls it at compile time) but cannot iterate a plain
    Python list of modules, nor zip() two ModuleLists together -- so pairing the
    dconv with its dsamp has to happen inside a module rather than in the loop.
    '''
    def __init__(self, dc, ds):
        super().__init__()
        self.dconv = dc
        self.dsamp = ds

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        over = self.dconv(x)
        return over, self.dsamp(over)


class _UpBlock(nn.Module):
    '''
    One rung of the upward leg: the umerge joining the skip with the rung below,
    followed by that rung's dconv.  See _DownBlock for why this is a module.
    '''
    def __init__(self, m, dc):
        super().__init__()
        self.umerge = m
        self.dconv = dc

    def forward(self, over: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return self.dconv(self.umerge(over, up))


def make_dconv(ich, factor, padding=False, batch_norm=False):
    n_padding = 1 if padding else 0
    och = ich
    if ich < 64:
        och = 64  # special first case
    elif factor != 1:
        och = int(ich*factor)
    node = dconv(ich, och, padding=n_padding, batch_norm=batch_norm)
    return node, och

def make_dsamp(ich):
    return dsamp(), ich

def make_umerge(ich, bilinear=False, padding=False):
    '''
    ich is number of channels from the skip
    '''
    # Assume umerge halves the number of channels from the input below.
    och = ich * 2
    return umerge(2*ich, bilinear=bilinear, padding=padding), och



class UNet(nn.Module):
    '''
    U-Net model exactly as from the paper by default.
    '''
    def _add_node(self, name, node):
        '''
        Add a subgraph node as an attribute on self.
        '''
        assert not hasattr(self, name)
        # print(f'node: {name} {node}')
        setattr(self, name, node)

    def __init__(self, n_channels=3, n_classes=6, in_shape=(572,572),
                 nskips=4,
                 batch_norm=False, bilinear=False, padding=False):
        super().__init__()
                
        nch = n_channels
        self.nskips = nskips

        # The legs are nn.ModuleLists rather than the flat down_dconv_<i> /
        # up_umerge_<i> attributes they used to be, so that forward_features can
        # iterate them under TorchScript.  Construction order is unchanged, which
        # keeps the *positional* parameter order identical -- an optimizer state
        # dict from before this change references parameters by position, so
        # reordering here would silently misplace momentum buffers on resume.
        # State-dict *key* names do change; see _remap_legacy_keys().
        down_blocks = list()
        skip_nchannels = list()      # for making skips
        for iskip in range(nskips):  # go down the U making dconv and dsamp

            dc_node, nch = make_dconv(nch, factor=2, padding=padding, batch_norm=batch_norm)
            skip_nchannels.append(nch)

            ds_node, nch = make_dsamp(nch)

            down_blocks.append(_DownBlock(dc_node, ds_node))
        self._add_node("down_blocks", nn.ModuleList(down_blocks))

        factor = 1 if padding else 2
        bottom, nch = make_dconv(nch, factor=factor, padding=padding, batch_norm=batch_norm)
        self._add_node("bottom", bottom)

        up_blocks = list()
        for iskip in range(nskips-1, -1, -1):   # bottom up order

            nch = skip_nchannels[iskip]
            m_node, nch = make_umerge(nch, bilinear=bilinear, padding=padding)

            factor = 0.25 if padding else 0.5
            dc_node, nch = make_dconv(nch, factor=factor, padding=padding, batch_norm=batch_norm)

            up_blocks.append(_UpBlock(m_node, dc_node))
        self._add_node("up_blocks", nn.ModuleList(up_blocks))

        self.out_features = nch  # channels of the pre-segmap feature map

        segmap = nn.Conv2d(nch, n_classes, 1)
        self._add_node("segmap", segmap)

        # Accept checkpoints written before the ModuleList restructuring.
        self._register_load_state_dict_pre_hook(self._remap_legacy_keys)


    def _remap_legacy_keys(self, state_dict, prefix, local_metadata, strict,
                           missing_keys, unexpected_keys, error_msgs):
        '''
        Rewrite, in place, state-dict keys written before the legs became
        nn.ModuleLists, so older checkpoints keep loading.

            down_dconv_<i>.*  ->  down_blocks.<i>.dconv.*
            down_dsamp_<i>.*  ->  down_blocks.<i>.dsamp.*
            up_umerge_<i>.*   ->  up_blocks.<nskips-1-i>.umerge.*
            up_dconv_<i>.*    ->  up_blocks.<nskips-1-i>.dconv.*

        bottom.* and segmap.* are unchanged.  The up leg is renumbered because it
        was built bottom-up: old index <i> counts from the top of the U, the new
        ModuleList position counts from the bottom.

        This is deliberately chatty: quietly renaming the keys of somebody's
        trained weights is exactly the kind of thing that should be visible.
        '''
        legs = (('down_dconv_', 'down_blocks.{}.dconv', False),
                ('down_dsamp_', 'down_blocks.{}.dsamp', False),
                ('up_umerge_',  'up_blocks.{}.umerge',  True),
                ('up_dconv_',   'up_blocks.{}.dconv',   True))

        renames = dict()
        nunder = 0
        for key in list(state_dict.keys()):
            if not key.startswith(prefix):
                continue
            nunder += 1
            tail = key[len(prefix):]
            head, _, rest = tail.partition('.')
            for old, newfmt, flip in legs:
                if not head.startswith(old):
                    continue
                try:
                    idx = int(head[len(old):])
                except ValueError:
                    break
                if flip:
                    idx = self.nskips - 1 - idx
                newhead = newfmt.format(idx)
                renames[key] = prefix + newhead + ('.' + rest if rest else '')
                break

        if not renames:
            return                  # already the current layout

        label = 'UNet[%s]' % (prefix.rstrip('.') or 'root')
        log.info(f'{label}: legacy checkpoint layout detected, '
                 f'remapping {len(renames)} of {nunder} keys')
        groups = dict()
        for old, new in renames.items():
            ohead = old[len(prefix):].split('.')[0]
            nhead = '.'.join(new[len(prefix):].split('.')[:3])
            groups[(ohead, nhead)] = groups.get((ohead, nhead), 0) + 1
        for (ohead, nhead), cnt in sorted(groups.items()):
            log.info(f'{label}:   {ohead}.* -> {nhead}.*  ({cnt} keys)')
        if len(renames) < nunder:
            log.info(f'{label}:   {nunder - len(renames)} keys unchanged '
                     '(bottom.*, segmap.*)')

        for old, new in renames.items():
            state_dict[new] = state_dict.pop(old)

    @torch.jit.export
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        '''
        The forward pass up to but not including the final segmap 1x1 conv.
        '''
        overs: List[torch.Tensor] = list()
        for dblk in self.down_blocks:
            over, x = dblk(x)
            overs.append(over)

        x = self.bottom(x)

        # up_blocks runs bottom-up, so it pairs with overs walked backwards.  A
        # reversed()/zip() over the ModuleList is not scriptable, hence the index.
        iover = len(overs) - 1
        for ublk in self.up_blocks:
            x = ublk(overs[iover], x)
            iover -= 1

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.segmap(self.forward_features(x))
