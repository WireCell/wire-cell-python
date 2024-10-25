#!/usr/bin/env python3

import click
from wirecell.util.cli import context, log, jsonnet_loader

@context("dnn")
def cli(ctx):
    '''
    Wire Cell Deep Neural Network commands.
    '''
    pass

@cli.command('train-dnnroi')
@click.option("-c", "--config",
              type=click.Path(),
              help="Set configuration file")
@click.pass_context
def train_dnnroi(ctx, config):
    '''
    Train the DNNROI model.
    '''
    log.info(config)

@cli.command("vizmod")
@click.option("-s","--size", default=572, help="Number of input pixel (rows or columns)") 
@click.option("-c","--channels", default=3, help="Number of input image channels") 
@click.option("-C","--classes", default=6, help="Number of output classes") 
@click.option("-b","--batch", default=1, help="Number of batch images") 
@click.option("-o","--output", default=None, help="File name to fill with GraphViz dot") 
@click.option("-m","--model", default="UNet",
              type=click.Choice(["UNet","UsuyamaUNet", "MilesialUNet","list"]))
def vizmod(size, channels, classes, batch, output, model):
    '''
    Produce a text summary and if -o/--output given also a GraphViz diagram of a named model.
    '''
    import torch
    from wirecell.dnn import models

    if model == "list":
        for one in dir(models):
            if one[0].isupper():
                print(one)
        return

    Mod = getattr(models, model)

    mod = Mod(channels, classes, size)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    mod = mod.to(device)

    from torchsummary import summary

    shape = (channels, size, size)
    summary(mod, input_size=shape, device=device)

    if output:
        from torchview import draw_graph
        shape = (batch, channels, size, size)
        gr = draw_graph(mod, input_size=shape, device=device)
        with open(output, "w") as fp:
            fp.write(str(gr.visual_graph))


def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
