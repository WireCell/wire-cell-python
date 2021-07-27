#!/usr/bin/env python
'''
Some general purpose plotting helpers
'''
import os

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


class NameSequence(object):
    def __init__(self, name, first=0):
        '''
        Every time called, emit a new name with an index.

        Name may be a filename with .-separated extension.

        The name may include zero or one %-format mark which will be
        fed an integer counting the number of output calls.

        If no format marker then the index marker is added between
        file base and extension (if any).

        The first may give the starting index sequence or if None no
        sequence will be used and the name will be kept as-is.

        This is a callable and it mimics PdfPages.
        '''
        self.base, self.ext = os.path.splitext(name)
        self.index = first

    def __call__(self):
        if self.index is None:
            return self.base + self.ext

        try:
            fn = self.base % self.index
        except TypeError:
            fn = '%s%04d' % (self.base, self.index)
        self.index += 1
        return fn + self.ext

    def savefig(self, *args, **kwds):
        '''
        Act like PdfPages
        '''
        fn = self()
        plt.savefig(fn, **kwds)

    def __enter__(self):
        return self
    def __exit__(self, typ, value, traceback):
        return
        
        
