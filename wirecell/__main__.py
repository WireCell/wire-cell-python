#!/usr/bin/env python

import click
from importlib import import_module

subs = "sigproc util gen pgraph resp plot aux ls4gan validate img test dnn pytorch"



@click.group()
def cli():
    """Main wcpy"""

for sub in subs.split():
    try:
        mod = import_module(f'wirecell.{sub}.__main__')
    except ModuleNotFoundError:
        continue
    cli.add_command(mod.cli)
    


def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
