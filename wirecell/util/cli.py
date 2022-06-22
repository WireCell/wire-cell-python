#!/usr/bin/env python3
'''
Some helpers for __main__.py CLIs
'''

import click
import functools

# use like:
#
# from wirecell.util.cli import jsonnet_loader
# from wirecell.util import jsio
# 
# @click.command()
# @jsonnet_loader
# @click.argument("filename")
# def mycmd(filename, **kwds):
#     jsio.load(filename, **kwds)

def jsonnet_loader(func):
    @click.option("-J", "--jpath", multiple=True,
                  envvar='WIRECELL_PATH', 
                  help="A dot-delimited path into the JSON to locate a graph-like object")
    @click.option("-A", "--tla", multiple=True,
                  help="Set a top-level argument as key=val, key=code or key=filename")
    @click.option("-V", "--ext", multiple=True,
                  help="Set an external var (avoid this with new jsonnet code)")
    @functools.wraps(func)
    def wrapper(*args, **kwds):
        return func(*args, **kwds)
    return wrapper
