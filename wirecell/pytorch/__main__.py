#!/usr/bin/env python3
'''
Main CLI for things related to wire-cell-toolkit/pytorch/ 
'''

import click
import torch

@click.group()
@click.pass_context
def cli(ctx):
    '''
    wirecell-pytorch command line interface
    '''
    ctx.ensure_object(dict)


@cli.command("make-dft")
@click.argument("filename")
def make_dft(filename):
    '''
    Generate the DFT torch script module into given file.
    '''
    from .script import DFT
    ts = torch.jit.script(DFT());
    ts.save(filename)

def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
