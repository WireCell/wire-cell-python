#!/usr/bin/env python3

import click
from wirecell.util.cli import context, log, jsonnet_loader
from wirecell.util.paths import unglob

@context("dnn")
def cli(ctx):
    '''
    Wire Cell Deep Neural Network commands.
    '''
    pass

@cli.command('train')
@click.option("-c", "--config",
              type=click.Path(),
              help="Set configuration file")
@click.argument("files", nargs=-1)
@click.pass_context
def train_dnnroi(ctx, config, files):
    '''
    Train the DNNROI model.
    '''
    fpaths = unglob(files)
    print (fpaths)

    log.info(config)
    from wirecell.dnn.apps import dnnroi



@cli.command("vizmod")
@click.option("-s","--shape", default="572,572",
              help="2D shape of input image in pixels") 
@click.option("-c","--channels", default=3, help="Number of input image channels") 
@click.option("-C","--classes", default=6, help="Number of output classes") 
@click.option("-b","--batch", default=1, help="Number of batch images") 
@click.option("-o","--output", default=None, help="File name to fill with GraphViz dot") 
@click.option("-m","--model", default="UNet",
              type=click.Choice(["UNet","UsuyamaUNet", "MilesialUNet","list"]))
def vizmod(shape, channels, classes, batch, output, model):
    '''
    Produce a text summary and if -o/--output given also a GraphViz diagram of a named model.
    '''
    import torch
    from wirecell.dnn import models

    imshape = shape.split(",")
    if len(imshape) == 1:
        imshape = [imshape, imshape]
    imshape = tuple(map(int, imshape))

    if model == "list":
        for one in dir(models):
            if one[0].isupper():
                print(one)
        return

    Mod = getattr(models, model)

    mod = Mod(channels, classes, imshape)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    mod = mod.to(device)

    from torchsummary import summary

    full_shape = (channels, imshape[0], imshape[0])
    summary(mod, input_size=full_shape, device=device)

    if output:
        from torchview import draw_graph
        batch_shape = (batch, channels, imshape[0], imshape[0])
        gr = draw_graph(mod, input_size=bach_shape, device=device)
        with open(output, "w") as fp:
            fp.write(str(gr.visual_graph))


def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
