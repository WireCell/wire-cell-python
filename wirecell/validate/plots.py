#!/usr/bin/env python
'''
Make some validation plots
'''
import numpy
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def make_cmaps():

    soft0 = [
        (0.0, 1.0, 1.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
    ]

    # (relative point, low side color component, high side color component)

    hard0 = dict(
        red = (
            (0.0, 0.0, 0.0),
            (0.5, 0.0, 1.0),
            (1.0, 1.0, 1.0)
        ),
        green = (
            (0.0, 0.0, 1.0),
            (0.5, 0.0, 0.0),
            (1.0, 1.0, 1.0)
        ),
        blue= (
            (0.0, 0.0, 0.0),
            (0.5, 1.0, 0.0),
            (1.0, 0.0, 0.0)
        )
    )

    cm = LinearSegmentedColormap.from_list('wct_soft0', soft0, N=100)
    cm.set_bad(color='white')
    plt.register_cmap(cmap=cm)
    
    cm = LinearSegmentedColormap('wct_hard0', hard0, N=100)
    cm.set_bad(color='white')
    plt.register_cmap(cmap=cm)
make_cmaps()


def three_horiz(arrs, extents, name, baselines):
    '''
    Plot three arrays horizontally.
    '''
    #cmap = 'nipy_spectral'
    #cmap = 'RdBu_r'
    
    cmap = "wct_soft0"
    if name in "orig".split():
        cmap = "wct_hard0"
    
    zlabel = dict(orig="ADC", raw="nfADC", gauss="electrons", wiener="electrons")[name]

    dpi = 100.0                 # nothing to do with actual DPI

    hpix = sum([a.shape[1] for a in arrs])
    vpix = a.shape[0]

    tpix = 60                   # title pix
    bpix = 60                   # border pix
    cpix = 100
    cibpix = 75                 # room for text for color bar

    blpix = 0
    if baselines:               # vertical room for baseline plots
        blpix = 100

    width_pix = hpix + len(arrs)*bpix*2 + cpix
    width_inch = width_pix/dpi
    height_pix = vpix + 2*bpix + blpix + tpix
    height_inch = height_pix/dpi

    bot_pix = blpix + bpix      # bottom of main event plots

    figsize=(width_inch, height_inch)
    #print "figsize=",figsize
    fig = plt.figure(dpi=dpi, figsize = figsize)

    ims = list()
    axes = list()

    left_pix = 0
    vmin = min([numpy.min(arr) for arr in arrs])
    vmax = max([numpy.max(arr) for arr in arrs])
    #print "vmm=",vmin,vmax
    vmm = max([abs(vmin), abs(vmax)])
    vmin = -vmax
    vmax =  vmax

    x_axes = list()

    for letter, arr, ext in zip("UVW", arrs, extents):
        tit = "%d ticks x %d ch grps" % (arr.shape[0], arr.shape[1])

        xa = (left_pix+bpix)/float(width_pix) # left
        ya = bot_pix/float(height_pix)        # bottom
        dxa = arr.shape[1]/float(width_pix)   # width
        dya = arr.shape[0]/float(height_pix)  # height
        relaxis = (xa,ya,dxa,dya)
        x_axes.append((xa,dxa))

        left_pix += 2*bpix + arr.shape[1] # move for next time

        #print "axis=",relaxis
        ax = fig.add_axes(relaxis)
        axes.append(ax)
        ax.set_title(tit)
        ax.set_xlabel("%s channels" % letter)
        ax.set_ylabel("ticks")

        im = ax.imshow(arr, cmap=cmap, interpolation='none', extent=ext, aspect='auto', vmin=vmin, vmax=vmax)
        ims.append(im)


    relaxis = [(left_pix)/float(width_pix),
               bot_pix/float(height_pix),
               (cpix-cibpix)/float(width_pix),
               (arrs[0].shape[0])/float(height_pix)]
    cbar_ax = fig.add_axes(relaxis)
    fig.colorbar(ims[0], ax=axes[0], cmap=cmap, cax=cbar_ax, label=zlabel)
            

    if not baselines:
        return fig
    
    blaxes = list()
    left_pix = 0
    for letter, blarr, ext, (xa,dxa) in zip("UVW", baselines, extents, x_axes):

        relaxis = (xa,
                   bpix/float(height_pix),           # bottom
                   dxa,
                   blpix/float(height_pix))          # height

        left_pix += 2*bpix + blarr.size/float(width_pix) # move for next time
            
        ax = fig.add_axes(relaxis)
        blaxes.append(ax)
        #ax.set_title(tit)
        ax.set_xlabel("%s channels" % letter)
        ax.set_ylabel("baseline")

        #print ext[:2], blarr.size
        ax.plot(numpy.linspace(ext[0], ext[1], num=blarr.size), blarr)
        ax.set_xlim([ext[0],ext[1]]) # really do what I say

    return fig



def one_plane(arr, extent, name):
    cmap = 'nipy_spectral'

    dpi = 100                   # nothing to do with actual DPI
    size = (arr.shape[0]/dpi, arr.shape[1]/dpi)

    fig = plt.figure(dpi=dpi, figsize=size)
    ax = fig.add_subplot(1,1,1)
    ax.set_title(name)
    im = ax.imshow(arr, cmap=cmap, interpolation='none', extent=extent, aspect='auto')
    fig.colorbar(im, cmap=cmap)
    
    return fig
    

def channel_summaries(name, arrs, extents):
    import matplotlib
    matplotlib.rcParams.update({'font.size': 8})

    mnames = arrs.keys()
    mnames.sort()
    
    fig, axes = plt.subplots(len(mnames), len(arrs[mnames[0]]), figsize=(8.5,11), dpi=100)
    for irow, mname in enumerate(mnames):
        for icol, (arr,letter) in enumerate(zip(arrs[mname],"UVW")):
            ext = extents[icol]
            ls = numpy.linspace(ext[0], ext[1], num=arr.size)
            ax = axes[irow,icol]
            ax.plot(ls, arr)
            ax.set_xlim([ext[0],ext[1]]) # really do what I say
            ax.set_title("%s-plane %s '%s'" % (letter, mname, name));
            #ax.set_xlabel("%s chans" % (letter, ))
    fig.tight_layout()

    return fig

                    
