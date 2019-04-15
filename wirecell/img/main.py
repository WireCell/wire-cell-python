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
    from . import converter, tap
    from tvtk.api import write_data
    
    gr = tap.load(cluster_tap_file)
    if 0 == gr.number_of_nodes():
        click.echo("no verticies in %s" % cluster_tap_file)
        return
    dat = converter.clusters2blobs(gr)
    write_data(dat, paraview_file)
    click.echo(paraview_file)
    return

@cli.command("paraview-activity")
@click.argument("cluster-tap-file")
@click.argument("paraview-file")
@click.pass_context
def paraview_activity(ctx, cluster_tap_file, paraview_file):
    '''
    Convert a JsonClusterTap JSON file to a ParaView .vti file of activity
    '''
    from . import converter, tap
    from tvtk.api import write_data
    
    gr = tap.load(cluster_tap_file)
    if 0 == gr.number_of_nodes():
        click.echo("no verticies in %s" % cluster_tap_file)
        return
    alldat = converter.clusters2views(gr)
    for wpid,dat in alldat.items():
        fname = paraview_file%wpid
        write_data(dat, fname)
        click.echo(fname)
    return


@cli.command("paraview-depos")
@click.argument("depo-npz-file")
@click.argument("paraview-file")
@click.pass_context
def paraview_blobs(ctx, depo_npz_file, paraview_file):
    '''
    Convert an NPZ file to a ParaView .vtu file of depos
    '''
    from . import converter
    from tvtk.api import write_data
    import numpy
    
    fp = numpy.load(open(depo_npz_file))
    dat = fp['depo_data_0']
    ugrid = converter.depos2pts(dat);
    write_data(ugrid, paraview_file)
    click.echo(paraview_file)
    return

def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
    
