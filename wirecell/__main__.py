#!/usr/bin/env python

import click
from importlib import import_module

subs = "sigproc util gen pgraph resp plot aux ls4gan validate img test dnn pytorch bee raygrid"

from wirecell.util.cli import context, log

@context("wirecell")
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


@cli.command("summary")
@click.option("-w", "--width", default=78, help="Line width for help text [default: 78]")
@click.pass_context
def cmd_summary(ctx, width):
    """Print a one-line summary of every wcpy namespace and its commands."""
    root = ctx.find_root().command
    cmd_col = 32          # width of the command-name column
    help_width = max(20, width - cmd_col - 4)
    for ns_name, ns_cmd in sorted(root.commands.items()):
        if ns_name == "summary":
            continue
        ns_help = ns_cmd.get_short_help_str(limit=width)
        click.echo(f"\n{ns_name}  {ns_help}")
        if hasattr(ns_cmd, "commands"):
            for cmd_name, cmd in sorted(ns_cmd.commands.items()):
                short = cmd.get_short_help_str(limit=help_width)
                click.echo(f"  {cmd_name:<{cmd_col}} {short}")


def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
