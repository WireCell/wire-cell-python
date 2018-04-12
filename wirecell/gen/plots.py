#!/usr/bin/env python

'''
Make some plots from simulation output
'''

import numpy
import matplotlib.pyplot as plt

def numpy_saver(filename, outfile):
    '''
    plot data from Numpy*Saver files
    '''
    tick1=0
    tick2=500

    fp = numpy.load(filename)
    frame = fp['frame__0'].T
    channels = fp['channels__0']
    chd = channels[1:] - channels[:-1]
    chjumps = [0] + list(numpy.where(chd>1)[0]) + [channels.size-1]
    indjumps = zip(chjumps[:-1], chjumps[1:])

    njumps = len(indjumps)
    fig, axes = plt.subplots(nrows=njumps, ncols=1)
    if njumps == 1:
        axes = [axes]

    for ax, (ind1, ind2) in zip(axes, indjumps):
        ch1 = channels[ind1+1]
        ch2 = channels[ind2]

        extent = (tick1, tick2, ch2, ch1)

        print ( "array ind: [%4d,%4d] channels: [%4d,%4d] %d" % (ind1,ind2, ch1, ch2, ind2-ind1+1))
        im = ax.imshow(frame[ind1+1:ind2,tick1:tick2],aspect='auto')#,extent=extent)
        plt.colorbar(im, ax=ax)

    plt.savefig(outfile)
    return fig,axes

if '__main__' == __name__:
    import sys
    numpy_saver(sys.argv[1], sys.argv[2])

