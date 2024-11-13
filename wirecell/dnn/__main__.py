#!/usr/bin/env python3

import click
import torch
from pathlib import Path
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
@click.option("-e", "--epochs", default=1,
              help="Number of epochs over which to train.  "
              "This is a relative count if the training starts with a -l/--load'ed state.")
@click.option("-b", "--batch", default=1, 
              help="Batch size")
@click.option("-d", "--device", default='cpu', 
              help="The compute device")
@click.option("--cache/--no-cache", is_flag=True, default=False,
              help="Cache data in memory")
@click.option("--debug-torch/--no-debug-torch", is_flag=True, default=False,
              help="Debug torch-level problems")
@click.option("--checkpoint-save", default=None,
              help="Checkpoint path.  "
              "An {epoch} pattern can be given to use the absolute epoch number")
@click.option("--checkpoint-modulus", default=1,
              help="Checkpoint modulus.  "
              "If checkpoint path is given, the training is checkpointed ever this many epochs..")
@click.option("-n", "--name", default='dnnroi',
              help="The application name (def=dnnroi)")
@click.option("-l", "--load", default=None,
              help="File name providing the initial model state dict (def=None - construct fresh)")
@click.option("-s", "--save", default=None,
              help="File name to save model state dict after training (def=None - results not saved)")
@click.argument("files", nargs=-1)
@click.pass_context
def train(ctx, config, epochs, batch, device, cache, debug_torch,
          checkpoint_save, checkpoint_modulus,
          name, load, save, files):
    '''
    Train a model.
    '''
    if not files:
        raise click.BadArgumentUsage("no files given")

    if device == 'gpu': device = 'cuda'
    log.info(f'using device {device}')

    if debug_torch:
        torch.autograd.set_detect_anomaly(True)

    # fixme: make choice of dataset optional
    import wirecell.dnn.apps
    from wirecell.dnn import io

    app = getattr(wirecell.dnn.apps, name)

    net = app.Network()
    opt = app.Optimizer(net.parameters())

    par = dict(epoch=0, loss=0)

    if load:
        if not Path(load).exists():
            raise click.FileError(load, 'warning: DNN module load file does not exist')
        par = io.load_checkpoint(load, net, opt)

    tot_epoch = par["epoch"]
    del par

    ds = app.Dataset(files, cache=cache)
    nsamples = len(ds)
    if nsamples == 0:
        raise click.BadArgumentUsage(f'no samples from {len(files)} files')

    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=batch, shuffle=True, pin_memory=True)
 
    trainer = app.Trainer(net, device=device)

    checkpoint=2                # fixme make configurable
    for epoch in range(epochs):
        losslist = trainer.epoch(dl)
        loss = sum(losslist)
        log.debug(f'epoch {tot_epoch} loss {loss}')

        if checkpoint_save:
            if tot_epoch%checkpoint_modulus == 0:
                cpath = checkpoint_save.format(epoch=tot_epoch)
                io.save_checkpoint(cpath, net, opt, 
                                   epoch=tot_epoch, loss=loss)
        tot_epoch += 1

    if save:
        io.save_checkpoint(save, net, opt, epoch=tot_epoch, loss=loss)


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

    log.info(f'dataset has {len(ds)} entries from {len(datapaths)} data paths')

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
                log.info(one)
        return

    Mod = getattr(models, model)

    log.info(f'{channels=} {classes=} {imshape=} {skips=} {batchnorm=} {bilinear=} {padding=}')

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
