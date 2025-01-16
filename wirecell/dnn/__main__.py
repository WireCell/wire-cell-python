#!/usr/bin/env python3

import time
import click
import torch
from torch.utils.data import DataLoader

from pathlib import Path
from wirecell.util.cli import context, log, jsonnet_loader, anyconfig_file
from wirecell.util.paths import unglob, listify


from wirecell import dnn


@context("dnn")
def cli(ctx):
    '''
    Wire Cell Deep Neural Network commands.
    '''
    pass

@cli.command('dump-config')
@anyconfig_file("wirecelldnn")
@click.pass_context
def dump_config(ctx, config):
    print(config)

    return


train_defaults = dict(epochs=1, batch=1, device='cpu', name='dnnroi', train_ratio=0.8)
@cli.command('train')
@click.option("-e", "--epochs", default=None, type=int,
              help="Number of epochs over which to train.  "
              "This is a relative count if the training starts with a -l/--load'ed state.")
@click.option("-b", "--batch", default=None, type=int,
              help="Batch size")
@click.option("-d", "--device", default=None, type=str,
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
@click.option("-a", "--app", default=None, type=str,
              help="The application name")
@click.option("-l", "--load", default=None,
              help="File name providing the initial model state dict (def=None - construct fresh)")
@click.option("-s", "--save", default=None,
              help="File name to save model state dict after training (def=None - results not saved)")
@click.option("--train-ratio", default=None, type=float,
              help="Fraction of samples to use for training (default=1.0, no evaluation loss calculated)")
@anyconfig_file("wirecelldnn", section='train', defaults=train_defaults)
@click.argument("files", nargs=-1)
@click.pass_context
def train(ctx, config, epochs, batch, device, cache, debug_torch,
          checkpoint_save, checkpoint_modulus,
          app, load, save, train_ratio, files):
    '''
    Train a model.
    '''

    if not files:               # args not processed by anyconfig_files
        try:
            files = config['train']['files']
        except KeyError:
            files = None
    if not files:
        raise click.BadArgumentUsage("no training files given")
    files = unglob(listify(files))
    log.info(f'training files: {files}')

    if device == 'gpu': device = 'cuda'

    if debug_torch:
        torch.autograd.set_detect_anomaly(True)

    name = app
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

    ds_dt = time.time()
    ds = app.Dataset(files, cache=cache, config=config.get("dataset", None))
    if len(ds) == 0:
        raise click.BadArgumentUsage(f'no samples from {len(files)} files')
    ds_dt = time.time() - ds_dt
    log.debug(f'Create dataset in {ds_dt:.3e} s')

    tbatch,ebatch = batch,1

    dses = dnn.data.train_eval_split(ds, train_ratio)
    dles = [DataLoader(one, batch_size=bb, shuffle=True, pin_memory=True) for one,bb in zip(dses, [tbatch,ebatch])]
            
    ntrain = len(dses[0])
    neval = len(dses[1])

    # History
    run_history = history.get("runs", dict())
    this_run_number = 0
    if run_history:
        this_run_number = max(run_history.keys()) + 1
    this_run = dict(
        run = this_run_number,
        data_files = files,
        ntrain = ntrain,
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

        train_loss = 0
        train_losses = []
        dt=0
        if ntrain:
            dt = time.time()
            train_losses = trainer.epoch(dles[0])
            train_loss = sum(train_losses)/ntrain
            dt = time.time() - dt

        eval_loss = 0
        eval_losses = []
        if neval:
            eval_losses = trainer.evaluate(dles[1])
            eval_loss = sum(eval_losses) / neval

        this_epoch = dict(
            run=this_run_number,
            epoch=this_epoch_number,
            train_losses=train_losses,
            train_loss=train_loss,
            eval_losses=eval_losses,
            eval_loss=eval_loss)
        epoch_history[this_epoch_number] = this_epoch

        log.info(f'run: {this_run_number} epoch: {this_epoch_number} loss: {train_loss:.4e} [b={tbatch},n={ntrain}] eval: {eval_loss:.4e} [b={ebatch},n={neval}] {dt=:.3e} s')

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
