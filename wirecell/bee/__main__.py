#!/usr/bin/python3
'''
Commands to work with Bee server and files.
'''

import json
import click
import numpy

from wirecell.util.cli import context, log
from .ana import summarize_series
from .data import load as load_bee

@context("bee")
def cli(ctx):
    '''
    Wire Cell Bee helpers
    '''
    pass


@cli.command("summary")
@click.argument("files", nargs=-1)
@click.pass_context
def cmd_summary(ctx, files):
    '''
    Print summary of Bee files (.zip or .json).
    '''
    for fname in files:
        series = load_bee(fname)
        text = summarize_series(series)
        print(f'{fname}:\n{text}')


@cli.command("diff")
@click.argument("files", nargs=2)
@click.pass_context
def cmd_diff(ctx, files):
    '''
    Diff two Bee files (pair of .zip or .json).

    This analyzes the point clouds and clusters.
    '''
    pass

def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
