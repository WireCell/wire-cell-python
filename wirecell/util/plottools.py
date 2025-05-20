#!/usr/bin/env python
'''
Some general purpose plotting helpers
'''
import os

from pathlib import Path

def rescaley(ax, x, y, rx, extra=0.1):
    '''
    Set ylim of ax so that the range rx of x is scaled to the view.

    The extra is an amount relative to the nominally resulting ylim to pad.
    '''
    import numpy
    inview = numpy.where( (x > rx[0]) & (x < rx[1]) )[0]
    ymin = y[inview].min()
    ymax = y[inview].max()
    dy = ymax-ymin
    lim = (ymin-(extra*dy), ymax+extra*dy)
    ax.set_ylim( *lim )
    return lim


class NameSequence(object):

    def __init__(self, name, first=0, **kwds):
        '''
        Every time called, emit a new name with an index.

        Name may be a filename with .-separated extension.

        The name may include zero or one %-format mark which will be
        fed an integer counting the number of output calls.

        If no format marker then the index marker is added between
        file base and extension (if any).

        The first may give the starting index sequence or if None no
        sequence will be used and the name will be kept as-is.

        Any keywords will be applied to savefig().

        This is a callable and it mimics PdfPages.
        '''
        self.base, self.ext = os.path.splitext(name)
        self.index = first
        self.opts = kwds

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
        import matplotlib.pyplot as plt

        opts = dict(self.opts, **kwds)

        fn = self()
        dirn = os.path.dirname(fn)
        if dirn and not os.path.exists(dirn):
            os.makedirs(dirn)
        plt.savefig(fn, **opts)


    def __enter__(self):
        return self
    def __exit__(self, typ, value, traceback):
        return
        
        
class NameSingleton(object):

    def __init__(self, path, **kwds):
        '''
        Like a NameSequence but force a singleton.

        No name mangling, and subsequent calls are ignored.

        Any kwds are applied to savefig.

        '''
        self.path = Path(path)
        self.called = 0
        self.opts = kwds

    def __call__(self):
        return self.path

    def savefig(self, *args, **kwds):
        '''
        Act like PdfPages
        '''
        import matplotlib.pyplot as plt

        opts = dict(self.opts, **kwds)

        if self.called == 0:
            if not self.path.parent.exists():
                self.path.parent.mkdir(parents=True)
            plt.savefig(self.path.absolute(), **opts)
        self.called += 1

    def __enter__(self):
        return self
    def __exit__(self, typ, value, traceback):
        return



def pages(name, format=None, single=False, **kwds):
    '''
    Return an instance of something like a PdfPages for the given format.

    Use like:

    >>> with pages(filename) as out:
    >>>    # make a matplotlib figure
    >>>    out.savefig()

    True multi-page formats (PDF) produce file with the given name.

    When a format that does not support pages (PNG) is requested then a page
    number is inserted into the file name.  The file name may be given with a
    '%d' template to explicitly describe how the page number should be set.
    Otherwise, the page number is appended to the base file name just before the
    file name extension.

    However, if "single" is True, then this numbering is not performed and each
    call to pages.savefig() will overwrite the file.
    '''

    if name.endswith(".pdf") or format=="pdf":
        from matplotlib.backends.backend_pdf import PdfPages
        return PdfPages(name, **kwds)
    if single:
        return NameSingleton(name, format)
    return NameSequence(name, **kwds)


def lg10(arr, eps = None, scale=None):
    '''
    Apply the "signed log" transform to an array.

    Result is +/-log10(|arr|*scale) with the sign of arr preserved in
    the result and any values that are in eps of zero set to zero.

    If eps is not given it is the smalles absolute value

    If scale is not given then 1/eps is used.
    '''
    import numpy
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

def image(array, style="image", fig=None, **kwds):
    '''
    Plot array as image, no axes.  Return result of imshow()
    '''
    import matplotlib.pyplot as plt

    if fig is None:
        fig = plt.figure(frameon=False)

    if style == "axes":
        kwds['aspect'] = 'auto'
        im = plt.imshow(array, **kwds)
        plt.colorbar()
        plt.tight_layout()
        return im

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    return plt.imshow(array, **kwds)


def imopts(**kwds):
    '''
    Return subset of kwds which are relevant to imsave() type functions.
    '''
    ret = dict()
    for key in 'vmin vmax cmap'.split():
        if key in kwds:
            ret[key] = kwds[key]
            continue
    return ret

