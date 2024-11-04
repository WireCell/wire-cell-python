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


import torch
import torch.nn as nn
from torch.nn.functional import pad as nnpad


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
        self.pads = None
        self.slices = None
        if bilinear:
            self.upsamp = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.upsamp = nn.ConvTranspose2d(nchannels, nchannels//2, 2, stride=2)

    def forward(self, over, up):
        up = self.upsamp(up)

        if self.padding:
            # when not cropping we must pad special to match when target is odd size
            if self.pads is None:
                pad = list()
                for dim in [-1, -2]:
                    diff = over.shape[dim] - up.shape[dim]
                    half = diff // 2
                    pad += [half, diff-half]
                self.pads = tuple(pad)
            up = nnpad(up, self.pads)
        else:
            if self.slices is None:
                slices = [slice(None),] * len(up.shape)  # select all by default
                for dim in [-2,-1]:
                    hi = over.shape[dim]
                    lo = up.shape[dim]
                    if lo == hi:
                        continue
                    beg = (hi - lo) // 2
                    end = beg + lo
                    slices[dim] = slice(beg,end)
                print(f'{slices=}\n{over.shape=} {up.shape=}')
                self.slices = tuple(slices)
            over = over[self.slices]
            print(f'{over.shape=}')

        cat = torch.cat((over, up), dim=1)
        print(f'{cat.shape=}')
        return cat


def make_dconv(ich, factor, padding=False, batch_norm=False):
    n_padding = 1 if padding else 0
    och = ich
    if ich < 64:
        och = 64  # special first case
    elif factor != 1:
        och = int(ich*factor)
    print(f'dconv {ich=} {och=} {factor=} {padding=} {batch_norm=}')
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
    def __init__(self, n_channels=3, n_classes=6, in_shape=(572,572),
                 nskips=4,
                 batch_norm=False, bilinear=False, padding=False):
        super().__init__()
                
        nch = n_channels

        self.downleg = list()       # nodes in downward U leg
        skip_nchannels = list()      # for making skips
        for iskip in range(nskips):  # go down the U making dconv and dsamp

            dc_node, nch = make_dconv(nch, factor=2, padding=padding, batch_norm=batch_norm)
            setattr(self, f'down_dconv_{iskip}', dc_node)
            skip_nchannels.append(nch)

            ds_node, nch = make_dsamp(nch)
            setattr(self, f'down_dsamp_{iskip}', ds_node)

            self.downleg.append((dc_node, ds_node))

        factor = 1 if padding else 2
        self.bottom, nch = make_dconv(nch, factor=factor, padding=padding, batch_norm=batch_norm)

        # self.skips = list()
        self.upleg = list()
        for iskip in range(nskips-1, -1, -1):

            nch = skip_nchannels[iskip]
            m_node, nch = make_umerge(nch, bilinear=bilinear, padding=padding)
            setattr(self, f'up_umerge_{iskip}', m_node)

            factor = 0.25 if padding else 0.5
            dc_node, nch = make_dconv(nch, factor=factor, padding=padding, batch_norm=batch_norm)
            setattr(self, f'up_dconv_{iskip}', dc_node)

            self.upleg.append((m_node, dc_node))  # bottom up order

        self.segmap = nn.Conv2d(nch, n_classes, 1)
        
        
    def forward(self, x):
        dump(x, "in")

        overs = list()
        for skip, (dc,ds) in enumerate(self.downleg):
            x = dc(x)
            dump(x, f"down dc {skip}")
            overs.append(x)
            x = ds(x)
            dump(x, f"down ds {skip}")
        overs.reverse()
        x = self.bottom(x)
        dump(x, "bottom")
        for over, (m,d) in zip(overs, self.upleg):
            x = m(over, x)
            dump(x, "up merge")
            x = d(x)
            dump(x, "up dc")
            
        x = self.segmap(x)
        return x
def dump(x, msg=""):
    print(f'{x.shape} {msg}')
