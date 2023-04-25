#!/usr/bin/env python3
'''
Main CLI for things related to wire-cell-toolkit/pytorch/ 
'''

import sys
import click
from wirecell.util.cli import context, log

@context("pytorch")
def cli(ctx):
    '''
    wirecell-pytorch command line interface
    '''
    pass


@cli.command("make-dft")
@click.argument("filename")
def make_dft(filename):
    '''
    Generate the DFT torch script module into given file.
    '''
    try:
        import torch
    except ImportError:
        log.error("no torch module")
        sys.exit(1)

    from .script import DFT
    ts = torch.jit.script(DFT());
    ts.save(filename)

def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
