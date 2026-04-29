#!/usr/bin/env python

import click
from importlib import import_module

subs = "sigproc util gen pgraph resp plot aux ls4gan validate img test dnn pytorch bee raygrid docs"

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


def _ns_name_from_path(p):
    """Extract a top-level CLI namespace name from a slash or dot path.

    'wirecell/util', 'wirecell.util', and 'util' all return 'util'.
    """
    parts = p.replace('.', '/').strip('/').split('/')
    if parts[0] == 'wirecell' and len(parts) > 1:
        parts = parts[1:]
    return parts[0]


@cli.command("help")
@click.argument("paths", nargs=-1, metavar="PATH|all")
@click.option("-w", "--width", default=78, help="Line width for help text [default: 78]")
@click.pass_context
def cmd_help(ctx, paths, width):
    """Print a summary of wcpy namespaces and their commands.

    \b
    PATH may use slash or dot notation: wirecell/util or wirecell.util.
    Use 'all' to show every namespace.
    """
    if not paths:
        click.echo(ctx.get_help())
        ctx.exit()

    root = ctx.find_root().command
    cmd_col = 32
    help_width = max(20, width - cmd_col - 4)

    if len(paths) == 1 and paths[0] == 'all':
        ns_items = [(n, c) for n, c in sorted(root.commands.items()) if n != 'help']
    else:
        ns_items = []
        for p in paths:
            ns_name = _ns_name_from_path(p)
            if ns_name not in root.commands:
                raise click.BadParameter(f'unknown namespace: {ns_name!r}', param_hint='PATH')
            ns_items.append((ns_name, root.commands[ns_name]))

    for ns_name, ns_cmd in ns_items:
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
