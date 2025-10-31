#!/usr/bin/env python

import click
from importlib import import_module

subs = "sigproc util gen pgraph resp plot aux ls4gan validate img test dnn pytorch bee raygrid"

from wirecell.util.cli import log


@click.group()
def cli():
    """Main wcpy"""

errors = 0
for sub in subs.split():
    try:
        mod = import_module(f'wirecell.{sub}.__main__')
    except ModuleNotFoundError:
        log.warn(f'no cli for module: {sub}')
        errors += 1
        continue
    cli.add_command(mod.cli)
if errors:
    log.warn("""
    Missing CLIs are typically due to your environment lacking dependencies.
    Developers, consider doing:
    $ uv sync --all-extras
    Users, consider doing:
    $ uv run --with torch wcpy [...]
    """)
    
    


def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
