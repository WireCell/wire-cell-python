#!/usr/bin/env python3

import click
import torch
from pathlib import Path
from wirecell.util.cli import context, log, jsonnet_loader
from wirecell.util.paths import unglob, listify


from wirecell import dnn


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
@click.option("--eval-files", multiple=True, type=str, # fixme: remove this in favor of a single file set and a train/eval partitioning
              help="File path or globs as comma separated list to use for evaluation dataset")
@click.argument("train_files", nargs=-1)
@click.pass_context
def train(ctx, config, epochs, batch, device, cache, debug_torch,
          checkpoint_save, checkpoint_modulus,
          name, load, save, eval_files, train_files):
    '''
    Train a model.
    '''
    if not train_files:
        raise click.BadArgumentUsage("no training files given")
    train_files = unglob(listify(train_files))
    log.info(f'training files: {train_files}')

    if device == 'gpu': device = 'cuda'
    log.info(f'using device {device}')

    if debug_torch:
        torch.autograd.set_detect_anomaly(True)

    app = getattr(dnn.apps, name)

    net = app.Network()
    opt = app.Optimizer(net.parameters())
    crit = app.Criterion()
    trainer = app.Trainer(net, opt, crit, device=device)

    history = dict()
    if load:
        if not Path(load).exists():
            raise click.FileError(load, 'warning: DNN module load file does not exist')
        history = dnn.io.load_checkpoint(load, net, opt)

    train_ds = app.Dataset(train_files, cache=cache)
    ntrain = len(train_ds)
    if ntrain == 0:
        raise click.BadArgumentUsage(f'no samples from {len(train_files)} files')

    from torch.utils.data import DataLoader
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, pin_memory=True)
 
    neval = 0
    eval_dl = None
    if eval_files:
        eval_files = unglob(listify(eval_files, delim=","))
        log.info(f'eval files: {eval_files}')
        eval_ds = app.Dataset(eval_files, cache=cache)
        neval = len(eval_ds)
        eval_dl = DataLoader(train_ds, batch_size=batch, shuffle=False, pin_memory=True)
    else:
        log.info("no eval files")

    # History
    run_history = history.get("runs", dict())
    this_run_number = 0
    if run_history:
        this_run_number = max(run_history.keys()) + 1
    this_run = dict(
        run = this_run_number,
        train_files = train_files,
        ntrain = ntrain,
        eval_files = eval_files or [],
        neval = neval,
        nepochs = epochs,
        batch = batch,
        device = device,
        cache = cache,
        name = name,
        load = load,
    )
    run_history[this_run_number] = this_run

    epoch_history = history.get("epochs", dict())
    first_epoch_number = 0
    if epoch_history:
        first_epoch_number = max(epoch_history.keys()) + 1

    def saveit(path):
        if not path:
            return
        dnn.io.save_checkpoint(path, net, opt, runs=run_history, epochs=epoch_history)

    for this_epoch_number in range(first_epoch_number, first_epoch_number + epochs):
        train_losses = trainer.epoch(train_dl)
        train_loss = sum(train_losses)/ntrain

        eval_losses = []
        eval_loss = 0
        if eval_dl:
            eval_losses = trainer.evaluate(eval_dl)
            eval_loss = sum(eval_losses) / neval

        this_epoch = dict(
            run=this_run_number,
            epoch=this_epoch_number,
            train_losses=train_losses,
            train_loss=train_loss,
            eval_losses=eval_losses,
            eval_loss=eval_loss)
        epoch_history[this_epoch_number] = this_epoch

        log.info(f'run: {this_run_number} epoch: {this_epoch_number} loss: {train_loss} eval: {eval_loss}')

        if checkpoint_save:
            if this_epoch_number % checkpoint_modulus == 0:
                parms = dict(this_run, **this_epoch)
                cpath = checkpoint_save.format(**parms)
                saveit(cpath)
    saveit(save)


@cli.command('dump')
@click.argument("checkpoint")
@click.pass_context
def dump(ctx, checkpoint):
    '''
    Dump info about a checkpoint file.
    '''
    state = dnn.io.load_checkpoint_raw(checkpoint)
    for rnum, robj in state.get("runs",{}).items():
        print('run: {run} ntrain: {ntrain} neval: {neval}'.format(**robj))
    for enum, eobj in state.get("epochs",{}).items():
        print('run: {run} epoch: {epoch} train: {train_loss} eval: {eval_loss}'.format(**eobj))

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
