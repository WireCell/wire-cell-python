#!/usr/bin/env python3
'''
Some utility functions related to colormaps.
'''

def good(diverging=True, color=True):
    '''
    An opinionated selection of the available colormaps.

    Diverging picks a cmap with central value as white, o.w. zero is
    white.  If color is False then a grayscale is used.

    https://matplotlib.org/stable/tutorials/colors/colormaps.html
    '''
    if diverging:
        if color: return "seismic"
        return "seismic"        # no gray diverging?
    if color: return "Reds"
    return "Greys"
    
def tier(tier, color=True):
    '''
    Return a good color map for the given data tier
    '''
    for diverging in ('orig', 'raw'):
        if tier.startswith(diverging):
            return good_cmap(True, color)
    for sequential in ('gauss', 'wiener'):
        if tier.startswith(sequential):
            return good_cmap(False, color)
    return good_cmap()

