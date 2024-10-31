#!/usr/bin/env python
'''
The Ronneberger, Fischer and Brox U-Net by default.

https://arxiv.org/abs/1505.04597
https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png

Unlike several other implementations using the name "U-Net" that one runs
across, this one tries to exactly replicate what is in the paper, by default.

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

- skip :: "skip connection" (gray arrow), the center crop of dconv output and
  provides input to umerge.  A "skip level" counts the skips from 0 starting at
  the top of the U.  The "bottom" can be considered a skip level for purposes of
  calculating the output size of its dconv.

- umerge :: "up merge" (gray+green arrows), concatenation of the skip result
  with the usamp result and provides input to an dconv on the upward leg.

The default configuration produces U-Net.  The following optional extensions,
off by default, are supported:

- insert two BatchNorm2d in double convolution unit (dconv).
- use other than 4 skip connection levels.
- use non-square images data.

'''


import torch
import torch.nn as nn
from torch.nn.functional import grid_sample


def down_out_size(orig_size, skip_level):
    '''
    Return the output size from a down unit at a given skip level.

    Skip level counts from 0 transfer "over" the U via skip connection or bottom.

    '''
    size = orig_size
    for dskip in range(skip_level + 1):
        if dskip:
            size = size // 2
        size = size - 4
    return size

def up_in_size(orig_size, skip_level, nlevels = 4):
    '''
    Return the input size to an up unit (output of a skip) at a given skip level.

    The nlevels counts the number of skip connections across the U.
    '''
    size = down_out_size(orig_size, nlevels)
    for uskip in range( nlevels - skip_level):
        if uskip:
            size = size - 4
        size = size * 2
    return size


def dimension(in_channels = 1, in_size = 572, nskips = 4):
    '''
    Calculate 1D image channel and pixel dimensions for elements of the U-Net.

    - size :: the size of both input image dimensions (572 for U-Net paper).

    - nskips :: the number of skip connections (4 for U-Net paper)

    This returns four lists of size 2*nskips+1.  Each element of a list
    corresponds to one major "dconv" unit as we go along the U: nskips "down",
    one "bottom" and nskips "up".  The lists are:

    - number of input channels
    - number of output channels
    - input size
    - output size

    The [nskips] element refers to the bottom dconv.

    See skip_dimensions() to form similar lists from the output of this function
    for the skip connections.

    Note, the output segmentation map is excluded.  The final element in the
    lists refers to the top up dconv.
    '''
    chans_down_in = [in_channels] + [2**(6+n) for n in range(nskips)]  # includes bottom
    chans_down_out = [2**(6+n) for n in range(nskips+1)]
    chans_up_in = list(chans_down_out[1:])
    chans_up_in.reverse()
    chans_in = chans_down_in + chans_up_in
    chans_up_out = chans_down_in[1:]
    chans_up_out.reverse()
    chans_out = chans_down_out + chans_up_out

    size_in = [in_size]
    size_out = []
    for skip in range(nskips):
        siz = size_in[-1] - 4   # dconv reduction
        size_out.append(siz)
        size_in.append(siz // 2)  # max pool reduction
    size_out.append(size_in[-1] - 4)  # bottom out
    for rskip in range(nskips):
        size_in.append(size_out[-1] * 2)  # up conv
        size_out.append(size_in[-1] - 4)  # dconv reduction

    return (chans_in, chans_out, size_in, size_out)


def dimensions(in_channels = 1, in_shape = (572,572), nskips = 4):
    '''
    N-D version of dimension() where sizes are shapes.
    '''
    dims = [dimension(in_channels, in_size, nskips) for in_size in in_shape]
    in_chans = dims[0][0]
    out_chans = dims[0][1]
    in_shapes = tuple(zip(*[d[2] for d in dims]))
    out_shapes = tuple(zip(*[d[2] for d in dims]))
    return in_chans, out_chans, in_shapes, out_shapes


def skip_dimensions(dims):
    '''
    Reformat the output of dimensions() to the same form but for the skip
    connections in order of skip level.
    '''
    (chans_in, chans_out, shape_in, shape_out) = dims

    nskips = (len(chans_in)-1)//2

    schans_in = chans_out[:nskips]
    schans_out = schans_in      # skips preserve channel dim

    sshape_in = shape_out[:nskips]
    sshape_out = list(shape_in[-nskips:])
    sshape_out.reverse()
    sshape_out = tuple(sshape_out)
    return (schans_in, schans_out, sshape_in, sshape_out)


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
        parts.insert(3, nn.BatchNorm2d(out_ch))
        parts.insert(1, nn.BatchNorm2d(out_ch))

    return nn.Sequential(*parts)


def dsamp():
    '''
    The "down sampling".
    '''
    return nn.MaxPool2d(2)


def usamp(in_ch):
    '''
    The "up sampling".
    '''
    return nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
    # return nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


# def build_grid(source_size, target_size, batch_size = 1):
#     '''
#     Map output pixels to input pixels for cropping by grid_sample().

#     This assumes square images of given size.
#     '''
#     # simplified version of what is given in
#     # https://discuss.pytorch.org/t/cropping-a-minibatch-of-images-each-image-a-bit-differently/12247
#     k = float(target_size)/float(source_size)
#     direct = torch.linspace(-k,k,target_size).unsqueeze(0).repeat(target_size,1).unsqueeze(-1)
#     grid = torch.cat([direct,direct.transpose(1,0)],dim=2).unsqueeze(0)
#     return grid.repeat(batch_size, 1, 1, 1)

class skip(nn.Module):
    '''
    The "skip connection" providing a core cropping.
    '''
    def __init__(self, source_shape, target_shape, batch_size=1):
        super().__init__()
        self.crop = []
        for ssize, tsize in zip(source_shape, target_shape):
            margin = (ssize - tsize)//2
            self.crop.append (slice(margin, margin+tsize))

        # A fancier way to do it which, but why?
        # self.register_buffer('g', build_grid(source_size, target_size, batch_size))
        # grid should have shape: (nbatch, nrows, ncols, 2)

    def forward(self, x):
        # x must be (nbatch, nchannel, nrows, ncols)
        # print(f'grid: {self.g.shape} {self.g.dtype} {self.g.device}')
        # print(f'data: {x.shape} {x.dtype} {x.device}')
        # return grid_sample(x, self.g, align_corners=True, mode='nearest')
        return x[:,:,self.crop[0],self.crop[1]]


class umerge(nn.Module):
    '''
    The "upsample merge" of the outputs from a skip and a dconv.
    '''
    def __init__(self, nchannels):
        '''
        Give number of channels in the input to the upsampling port.
        '''
        super().__init__()
        self._nchannels = nchannels
        self.upsamp = nn.ConvTranspose2d(nchannels, nchannels//2, 2, stride=2)

    def forward(self, over, up):
        up = self.upsamp(up)
        cat = torch.cat((over, up), dim=1)
        return cat


class UNet(nn.Module):
    '''
    U-Net model exactly as from the paper by default.
    '''

    def __init__(self, n_channels=3, n_classes=6, in_shape=(572,572),
                 batch_size=1, nskips=4,
                 batch_norm=False):
        super().__init__()
                
        self.nskips = nskips
        dims = dimensions(n_channels, in_shape, nskips)

        # The major elements of the U
        chans_in, chans_out, _, _ = dims

        # Note; we use setattr to make sure PyTorch summary finds the submodules.

        # The downward leg of the U.
        for ind in range(nskips):
            setattr(self, f'downleg_{ind}', dconv(chans_in[ind], chans_out[ind]))

        # The bottom of the U
        self.bottom = dconv(chans_in[nskips], chans_out[nskips])

        # The upward leg of the U.
        for count, ind in enumerate(range(nskips+1, 2*nskips+1)):
            setattr(self, f'upleg_{count}', dconv(chans_in[ind], chans_out[ind]))

        # The skip connections get applied top-down
        schans_in, schans_out, ssize_in, ssize_out = skip_dimensions(dims)
        for ind, ss in enumerate(zip(ssize_in, ssize_out)):
            setattr(self, f'skip_{ind}', skip(*ss, batch_size=batch_size))

        # And the merges are applied bottom-up.
        # We bake in the rule that upsample input has 2x the number of channels as the skip output.
        umerges = [umerge(2*nc) for nc in schans_out]
        umerges.reverse()
        for ind, um in enumerate(umerges):
            setattr(self, f'umerge_{ind}', um);

        # Downsampler is data-independent and reused.
        self.dsamp = dsamp()

        self.segmap = nn.Conv2d(chans_out[-1], n_classes, 1)
        
    def getm(self, name, ind):
        return getattr(self, f'{name}_{ind}')
        
    def forward(self, x):

        dskips = list()

        for ind in range(self.nskips):
            dl = self.getm("downleg", ind)
            dout = dl(x)
            x = self.dsamp(dout)
            sm = self.getm("skip", ind)
            dskip = sm(dout)
            dskips.append( dskip )

        x = self.bottom(x)

        dskips.reverse()        # bottom-up
        for ind in range(self.nskips):
            s = dskips[ind]
            um = self.getm("umerge", ind)
            x = um(s, x)
            ul = self.getm("upleg", ind)
            x = ul(x)
            
        x = self.segmap(x)
        return x
