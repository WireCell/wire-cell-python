import math
import click

from wirecell.util import ario, plottools

from wirecell.util.cli import context, log

@context("test")
def cli(ctx):
    '''
    Wire Cell Test Commands
    '''
    pass

@cli.command("plot")
@click.option("-n", "--name", default="noise",
              help="The test name")
@click.argument("datafile")
@click.argument("output")
@click.pass_context
def plot(ctx, name, datafile, output):
    '''
    Make plots from file made by test_<test>.
    '''
    from importlib import import_module
    mod = import_module(f'wirecell.test.{name}')
    fp = ario.load(datafile)
    with plottools.pages(output) as out:
        mod.plot(fp, out)
    


def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
