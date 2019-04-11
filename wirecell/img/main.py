#!/usr/bin/env python
'''
The wirecell-img main
'''
import json
import click


@click.group("img")
@click.pass_context
def cli(ctx):
    '''
    Wire Cell Toolkit Imaging Commands
    '''
@cli.command("paraview-blobs")
@click.argument("cluster-tap-file")
@click.argument("paraview-file")
@click.pass_context
def paraview_blobs(ctx, cluster_tap_file, paraview_file):
    '''
    Convert a JsonClusterTap JSON file to a ParaView .vtu file of blobs
    '''
    from . import vtk
    from tvtk.api import write_data
    
    tap = json.load(open(cluster_tap_file))
    if 0 == len(tap["vertices"]):
        click.echo("no verticies in %s" % cluster_tap_file)
        return
    dat = vtk.clusters2blobs(tap)
    write_data(dat, paraview_file)
    click.echo(paraview_file)
    return


def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
    
