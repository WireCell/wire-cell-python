#!/usr/bin/env python
'''
Some general purpose plotting helpers
'''
import os

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy

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
            fn = self.base % (self.index,)
        except TypeError:
            fn = '%s%04d' % (self.base, self.index)
        self.index += 1

        ret = fn + self.ext
        return ret

    def savefig(self, *args, **kwds):
        '''
        Act like PdfPages
        '''
        fn = self()
        dirn = os.path.dirname(fn)
        if dirn and not os.path.exists(dirn):
            os.makedirs(dirn)
        plt.savefig(fn, **kwds)

    def __enter__(self):
        return self
    def __exit__(self, typ, value, traceback):
        return
        
        
def pages(name):
    if name.endswith(".pdf"):
        return PdfPages(name)
    return NameSequence(name)


def lg10(arr, eps = None, scale=None):
    '''
    Apply the "signed log" transform to an array.

    Result is +/-log10(|arr|*scale) with the sign of arr preserved in
    the result and any values that are in eps of zero set to zero.

    If eps is not given it is the smalles absolute value

    If scale is not given then 1/eps is used.
    '''
    if eps is None:
        eps = numpy.min(numpy.abs(arr))

    if not scale:
        scale = 1/eps

    shape = arr.shape
    arr = numpy.array(arr).reshape(-1)
    arr[numpy.logical_and(arr < eps, arr > -eps)] = 0.0
    pos = arr>eps
    neg = arr<-eps
    arr[pos] = numpy.log10(arr[pos]*scale)
    arr[neg] = -numpy.log10(-arr[neg]*scale)
    return arr.reshape(shape)
