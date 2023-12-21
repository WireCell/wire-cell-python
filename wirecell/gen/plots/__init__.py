#!/usr/bin/env python
'''
The wirecell.util.plots module
'''

# There used to be a plots.py module providing numpy_saver().
# This function is refactored but the old namespace retained. 
from .numpysaver import plot as numpy_saver
from . import morse
