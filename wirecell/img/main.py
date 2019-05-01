#!/usr/bin/env python
'''
The wirecell-img main
'''
import json
import click

from wirecell import units

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

#    Bee support:   
#    http://bnlif.github.io/wire-cell-docs/viz/uploads/


@cli.command("bee-blobs")
@click.option('-o', '--output', help="The output Bee JSON file name")
@click.option('-g', '--geom', default="protodune",
              help="The name of the detector geometry")
@click.option('--rse', nargs=3, type=int, default=[0, 0, 0],
              help="The '<run> <subrun> <event>' numbers as a triple of integers")
@click.option('-s', '--sampling', type=click.Choice(["center","uniform"]), default="uniform",
              help="The sampling technique to turn blob volumes into points")
@click.option('-d', '--density', type=float, default=9.0,
              help="For samplings which care, specify target points per cc")
@click.argument("cluster-tap-files", nargs=-1)
def bee_blobs(output, geom, rse, sampling, density, cluster_tap_files):
    '''
    Make one Bee JSON file from the blobs in a set of 'cluster tap'
    JSON files which are presumed to originate from one trigger.
    '''
    from . import tap, converter

    dat = dict(runNo=rse[0], subRunNo=rse[1], eventNo=rse[2], geom=geom, type="wire-cell",
               x=list(), y=list(), z=list(), q=list()) # , cluster_id=list()


    def fclean(arr):
        return [round(a, 3) for a in arr]
        
    # given by user in units of 1/cc.  Convert to system of units 1/L^3.
    density *= 1.0/(units.cm**3)
    sampling_func = dict(
        center = converter.blob_center,
        uniform = lambda b : converter.blob_uniform_sample(b, density),
    )[sampling];

    for ctf in cluster_tap_files:
        gr = tap.load(ctf)
        arr = converter.blobpoints(gr, sampling_func)
        print ("%s: %d points" % (ctf, arr.shape[0]))
        dat['x'] += fclean(arr[:,0]/units.cm)
        dat['y'] += fclean(arr[:,1]/units.cm)
        dat['z'] += fclean(arr[:,2]/units.cm)
        dat['q'] += fclean(arr[:,3])

    import json
    # monkey patch
    from json import encoder
    encoder.FLOAT_REPR = lambda o: format(o, '.3f')
    json.dump(dat, open(output,'w', encoding="utf8"))

def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
    
