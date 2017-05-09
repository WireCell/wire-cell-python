import click

from wirecell import units

@click.group("util")
@click.pass_context
def cli(ctx):
    '''
    Wire Cell Signal Processing Features
    '''

@cli.command("convert-oneside-wires")
@click.argument("input-file")
@click.argument("output-file")
@click.pass_context
def convert_oneside_wires(ctx, input_file, output_file):
    '''
    Convert a a "onesided" wires description file into one suitable for WCT.

    An example file is:

    https://github.com/BNLIF/wire-cell-celltree/blob/master/geometry/ChannelWireGeometry_v2.txt

    It has columns like:
    # channel plane wire sx sy sz ex ey ez

    The output file is JSON and if it has .gz or .bz2 it will be compressed.
    '''
    from wirecell.util.wires import onesided, persist
    store = onesided.load(input_file)
    persist.dump(output_file, store)


def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
    
