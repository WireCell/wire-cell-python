#!/usr/bin/env python
'''Utilities for dealing with arrays indices as "bounding boxes"

Here, a "bbox" is an object that may be used to index a numpy array, potentially
a multi-dimensional one.  

A bbox is represented as a sequence of ranges, one for each dimension of an
array.

A range may be in "slice form" or "array form".  A slice range is a slice()
instance.  An array range is any list-like sequence of integer indices.  Indices
may span the array dimension sparsely, out of order and may repeat.

Bbox functions may expand from slice to array form and may apply selection,
uniqueness and ordering to indices in an array form.

'''
import numpy

def union_array(*ranges, order="stack"):
    '''
    Form a union of ranges in array form.

    The union may be sparse and ordered.

    Order determines post-processing of the union.

    - "ascending" :: sort unique indices in ascending order.
    - "descending" :: sort unique indices in descending order.
    - "seen" value :: sort unique indices in first-seen order.
    - "stack" :: simply concatenate indices of each range (default).

    '''
    if not ranges:
        return numpy.array((0,))
    u = list()
    for one in ranges:
        u.append(numpy.r_[one])
    u = numpy.hstack(u)
    if order == "ascending":
        return numpy.unique(u)
    if order == "descending":
        u = numpy.unique(u)
        return u[::-1]
    if order == "seen":
        g,i = numpy.unique(u, return_index=True)
        return g[numpy.argsort(i)]

    return u                    # "stack" by default


def union_slice(*slices):
    '''Form union of ranges.

    A slice is returned that spans the union of the given slices.

    Input slices may define a .step value which is considered in forming the
    union.  In any case, the returned slice has no step defined.

    '''
    if not slices:
        return slice(None,None,None)
    inds = numpy.hstack([numpy.r_[s] for s in slices])
    return slice(numpy.min(inds), 1+numpy.max(inds))
    

def union(*bboxes, form="slices"):
    '''Form union of bboxes.

    If form is "slices" then the ranges of each dimension of the bboxes must all
    be slices and the returned bbox will have union of ranges formed with
    union_slice().  Otherwise, union_array() is used and "form" is passed as the
    "order".

    '''
    if form == "slices":
        return tuple([union_slice(*ranges) for ranges in zip(*bboxes)])
    return tuple([union_array(*ranges, order=form) for ranges in zip(*bboxes)])


# todo:
#
# - bounds(array) -> return smaller array that removes any rows/cols that are
# fully masked.  bonus to work on N-dimensions.  Surprisingly, such a function
# is not found in numpy, scipy?
