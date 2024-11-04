#!/usr/bin/env python3

import click

from wirecell.util.cli import context, log, jsonnet_loader
from wirecell.util.paths import unglob, listify


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
@click.option("-e", "--epochs", default=1, help="Number of epochs over which to train")
@click.option("-b", "--batch", default=1, help="Batch size")
@click.argument("files", nargs=-1)
@click.pass_context
def train(ctx, config, epochs, batch, files):
    '''
    Train the DNNROI model.
    '''
    # fixme: args to explicitly select use of "flow" tracking.
    from wirecell.dnn.tracker import flow

    # fixme: make choice of dataset optional
    from wirecell.dnn.apps import dnnroi as app



    # fixme: this should all be moved under the app 
    ds = app.Dataset(unglob(files))
    imshape = ds[0][0].shape[-2:]
    print(f'{imshape=}')

    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=batch, shuffle=True)

    net = app.Network(imshape, batch_size=batch)
    trainer = app.Trainer(net, tracker=flow)

    for epoch in range(epochs):
        loss = trainer.epoch(dl)
        flow.log_metric("epoch_loss", dict(epoch=epoch, loss=loss))

    # log.info(config)



@cli.command('extract')
@click.option("-o", "--output", default='samples.npz',
              help="Output in which to save the extracted samples")  # fixme: support also hdf
@click.option("-s", "--sample", multiple=True, type=str,
              help="Index or comma separated list of indices for samples to extract")
@click.argument("datapaths", nargs=-1)
@click.pass_context
def extract(ctx, output, sample, datapaths):
    '''
    Extract samples from a dataset.

    The datapaths name files or file globs.
    '''
    samples = map(int,listify(*sample, delim=","))

    # fixme: make choice of dataset optional
    from wirecell.dnn.apps import dnnroi as app
    ds = app.Dataset(datapaths)

    print(f'dataset has {len(ds)} entries from {len(datapaths)} data paths')

    # fixme: support npz and hdf and move this into an io module.
    import io
    import numpy
    import zipfile              # must diy to append to .npz
    from pathlib import Path
    with zipfile.ZipFile(output, 'w') as zf:
        for isam in samples:
            sam = ds[isam]
            for iten, ten in enumerate(sam):
                bio = io.BytesIO()
                numpy.save(bio, ten.cpu().detach().numpy())
                zf.writestr(f'sample_{isam}_{iten}.npy', data=bio.getbuffer().tobytes())


@cli.command('plot3p1')
@click.option("-o", "--output", default='samples.png',
              help="Output in which to save the extracted samples")  # fixme: support also hdf
@click.option("-s", "--sample", multiple=True, type=str,
              help="Index or comma separated list of indices for samples to extract")
@click.argument("datapaths", nargs=-1)
@click.pass_context
def plot4dnnroi(ctx, output, sample, datapaths):
    '''
    Plot 3 layers from first tensor and 1 image from second from each sample.
    '''

    samples = list(map(int,listify(*sample, delim=",")))

    # fixme: make choice of dataset optional
    from wirecell.dnn.apps import dnnroi as app
    ds = app.Dataset(datapaths)

    # fixme: move plotting into a dnn.plots module
    import matplotlib.pyplot as plt
    from wirecell.util.plottools import pages
    with pages(output, single=len(samples)==1) as out:

        for idx in samples:
            rec,tru = ds[idx]
            rec = rec.detach().numpy()
            tru = tru.detach().numpy()
            fig,axes = plt.subplots(2,2)
            axes[0][0].imshow(rec[0])
            axes[0][1].imshow(rec[1])
            axes[1][0].imshow(rec[2])
            axes[1][1].imshow(tru[0])

            out.savefig()


@cli.command("vizmod")
@click.option("-s","--shape", default="572,572",
              help="2D shape of input image in pixels") 
@click.option("-c","--channels", default=3, help="Number of input image channels") 
@click.option("-C","--classes", default=6, help="Number of output classes") 
@click.option("-b","--batch", default=1, help="Number of batch images") 
@click.option("--skips", default=4, help="Number skip layers") 
@click.option("--padding/--no-padding", default=False, is_flag=True, help="Use padding") 
@click.option("--bilinear/--no-bilinear", default=False, is_flag=True, help="Use bilinear upsampling") 
@click.option("--batchnorm/--no-batchnorm", default=False, is_flag=True, help="Use batch normalization") 
@click.option("-o","--output", default=None, help="File name to fill with GraphViz dot") 
@click.option("-m","--model", default="UNet",
              type=click.Choice(["UNet","UsuyamaUNet", "MilesialUNet","list"]))
def vizmod(shape, channels, classes, batch, skips, padding, bilinear, batchnorm, output, model):
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

    print(f'{channels=} {classes=} {imshape=} {skips=} {batchnorm=} {bilinear=} {padding=}')

    mod = Mod(channels, classes, imshape, nskips=skips,
              batch_norm=batchnorm, bilinear=bilinear, padding=padding)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    mod = mod.to(device)

    from torchsummary import summary

    full_shape = (channels, imshape[0], imshape[1])
    summary(mod, input_size=full_shape, device=device)

    if output:
        from torchview import draw_graph
        batch_shape = (batch, channels, imshape[0], imshape[1])
        gr = draw_graph(mod, input_size=batch_shape, device=device)
        with open(output, "w") as fp:
            fp.write(str(gr.visual_graph))


def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
