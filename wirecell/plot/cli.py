#!/usr/bin/env python3
'''

Decorators and functions for constructing Click commands.

'''

import click
import functools
from wirecell.util import ario
    
# Decorator for a CLI that is common to a couple commands.
#
# usage in a Click main context:
#
# @cli.command("my-frame-to-image-command-name")
# @frame_to_image
# def my_command_function(array, channels, cmap, format, output, aname, fname)
#
# Note, the args finally passed to the function are substantially processed.
# See existing usage in __main__.py for examples.
#
def frame_input(func):
    # import here hide dependencies from command that do not use this decorator.
    import numpy

    @click.option("-a", "--array", default="frame_*_0", help="array name")
    @click.option("-c", "--channels", default="800,800,960", help="comma list of channel counts per plane in u,v,w order")
    @click.argument("ariofile")
    @functools.wraps(func)
    def wrapper(*args, **kwds):

        channels = list(map(int, kwds["channels"].split(",")))
        if len(channels) != 3:
            raise click.BadParameter("must give 3 channel group counts")
        chrange = list()
        for i,c in enumerate(channels):
            if not i:
                chrange.append((0, c))
            else:
                l = chrange[i-1][1]
                chrange.append((l, l+c))
        kwds["channels"] = chrange
        
        fname = kwds.pop("ariofile")
        kwds['fname'] = fname
        # fp = numpy.load(fname)
        fp = ario.load(fname)
        
        aname = kwds.pop("array")
        if aname in fp:
            kwds["array"] = fp[aname]
        else:
            have = '", "'.join(fp.keys())
            raise click.BadParameter(f'array "{aname}" not in "{fname}".  try: "{have}"')
        kwds['aname'] = aname

        return func(*args, **kwds)
    return wrapper


def image_output(func):
    from matplotlib import colormaps

    @click.option("-C", "--cmap", default="gist_ncar", help="Color map name def=gist_ncar")
    @click.option("--vmin", default=None, help="Set min value")
    @click.option("--vmax", default=None, help="Set max value")
    @click.option("-f", "--format", default=None, help="Output file format, def=auto")
    @click.option("-o", "--output", default=None, help="Output file, def=stdout")
    @functools.wraps(func)
    def wrapper(*args, **kwds):
        fmt = kwds["format"]
        out = kwds["output"]
        if fmt is None:
            if out is None or len(out.split(".")) == 1:
                kwds["format"] = "png"
            else:
                kwds["format"] = out.split(".")[-1]
        if out is None:
            kwds["output"] = "/dev/stdout"

        kwds["cmap"] = colormaps[kwds["cmap"]]

        return func(*args, **kwds)
    return wrapper


