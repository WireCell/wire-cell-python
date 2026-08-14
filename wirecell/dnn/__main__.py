#!/usr/bin/env python3

import time
import click

from pathlib import Path
from wirecell.util.cli import context, log, jsonnet_loader, anyconfig_file
from wirecell.util.paths import unglob, listify


from wirecell import dnn

def obj_with_config(obj, config, key,  additional_args=[]):
    '''
    Build obj from the named config section, returning (instance, args).

    The section is passed as keyword arguments, so every app's Network takes
    **cfg even when it ignores it.  args is returned so callers can record the
    resolved configuration alongside a checkpoint.
    '''
    obj_config = config.get(key, None)
    args = ({} if obj_config is None else obj_config)
    log.debug(f'{key} config: {args}')
    return obj(*additional_args, **args), args


def resolve_model_config(cls, cfg, checkpoint):
    '''
    Let an app's Network reconcile its config section with a checkpoint being
    resumed from, returning the config to build the model with.

    What a resume must drop or agree on is model-specific -- which keys merely
    seed a fresh model, which fix parameter shapes or the training regime -- so
    a Network may declare a resolve_config() classmethod and own that decision.
    Networks that do not are left alone, so an app opts in just by defining it.
    '''
    resolver = getattr(cls, 'resolve_config', None)
    if checkpoint is None or resolver is None:
        return dict(cfg)
    args = dnn.io.checkpoint_model_args(checkpoint,
                                        getattr(cls, 'RECORDED_KEYS', ()))
    try:
        return resolver(dict(cfg), checkpoint_args=args)
    except ValueError as err:
        # A rejected resume is a config mistake, not a crash: the model classes
        # raise a plain ValueError so they stay independent of click, and it
        # becomes an ordinary "Error: ..." here rather than a traceback.
        raise click.ClickException(str(err)) from err

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
              help="Batch size.  Under DDP (torchrun) this is the per-GPU batch "
              "size, so the effective global batch is batch * nproc_per_node.")
@click.option("--eval-batch", default=None, type=int,
              help="Batch size for evaluation (default: same as --batch).  "
              "Eval runs in eval() mode so BatchNorm uses running stats; this "
              "only affects eval speed, not the metric.")
@click.option("-d", "--device", default=None, type=str,
              help="The compute device")
@click.option("--cache/--no-cache", is_flag=True, default=False,
              help="Cache data in memory")
@click.option("--amp/--no-amp", is_flag=True, default=False,
              help="Use mixed-precision (autocast) training.  "
              "Only takes effect on CUDA; a no-op on CPU.")
@click.option("--amp-dtype", default=None,
              type=click.Choice(['float16', 'bfloat16']),
              help="Autocast dtype for --amp (def=float16).  Use bfloat16 for "
              "models whose activations can overflow fp16's ~65504 range: it "
              "keeps fp32's exponent at the same speed.  Required for xvunet, "
              "where fp16 sends ~29% of batches to inf logits.")
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
@click.option("--manual-seed", default=None, type=int,
              help="Set this to use a manual torch seeding (default=None -> use default torch seeding)")
@click.option("--ddp-split-seed", default=0, type=int,
              help="Set this to use a manual seeding for the train/eval split -- only applicable under ddp")
@anyconfig_file("wirecelldnn", section='train', defaults=train_defaults)
@click.argument("files", nargs=-1)
@click.pass_context
def train(ctx, config, epochs, batch, eval_batch, device, cache, amp, amp_dtype,
          debug_torch, checkpoint_save, checkpoint_modulus,
          app, load, save, train_ratio, manual_seed, ddp_split_seed, files):
    '''
    Train a model.
    '''
    # delay importing this monster
    import torch
    if manual_seed is not None:
        torch.manual_seed(manual_seed)
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    import wirecell.dnn.apps

    # Initialize DDP if launched under torchrun (a no-op otherwise).  Everything
    # below runs per-rank; teardown happens in the finally.
    dnn.dist.setup()
    try:
        if not files:               # args not processed by anyconfig_files
            try:
                files = config['train']['files']
            except KeyError:
                files = None
        if not files:
            raise click.BadArgumentUsage("no training files given")
        files = unglob(listify(files))
        log.debug(f'training files: {files}')

        # Under DDP each rank binds its own GPU, overriding the --device string.
        if dnn.dist.is_dist():
            device = f'cuda:{dnn.dist.get_local_rank()}'
        elif device == 'gpu':
            device = 'cuda'

        if str(device).startswith('cuda'):
            # Input sizes are fixed per run, so let cuDNN autotune conv algorithms.
            torch.backends.cudnn.benchmark = True

        if debug_torch:
            torch.autograd.set_detect_anomaly(True)

        name = app
        app = getattr(wirecell.dnn.apps, name)

        # Read the checkpoint before building anything: resolving the model
        # config against it is what lets a resume skip the seed-only loads and
        # catch a regime mismatch here rather than as an opaque param-group
        # error from the optimizer.  Read once and reuse for the restore below --
        # these files run to hundreds of MB and every rank reads its own copy.
        ck = None
        if load:
            if not Path(load).exists():
                raise click.FileError(load, 'warning: DNN module load file does not exist')
            ck = dnn.io.load_checkpoint_raw(load)

        model_args = resolve_model_config(app.Network, config.get('model') or {}, ck)
        log.debug(f'model config: {model_args}')
        net = app.Network(**model_args)
        
        opt, _ = obj_with_config(app.Optimizer, config, 'optimizer', [net.parameters()])
        if dnn.dist.is_main():
            print(opt.state_dict())
        crit = app.Criterion()

        # [train] amp_dtype is honoured via the config section fill-in; default
        # fp16 keeps existing runs bit-for-bit unchanged.
        trainer = app.Trainer(net, opt, crit, device=device, amp=amp,
                              amp_dtype=(amp_dtype or 'float16'))
        if amp and dnn.dist.is_main():
            log.info(f'mixed precision enabled, autocast dtype '
                     f'{amp_dtype or "float16"}')

        history = dict()
        if ck is not None:
            history = dnn.io.load_checkpoint_from(ck, net, opt)
            # Both load_state_dict()s copy, so the file's own tensors are dead
            # weight from here on.  Drop them: ck is a local that would otherwise
            # live for the whole run, holding a second copy of every parameter
            # and optimizer moment -- per rank, under DDP.
            ck = None

        ds_dt = time.time()
        ds = app.Dataset(files, cache=cache, config=config.get("dataset", None))
        if len(ds) == 0:
            raise click.BadArgumentUsage(f'no samples from {len(files)} files')
        ds_dt = time.time() - ds_dt
        log.debug(f'Create dataset in {ds_dt:.3e} s')

        tbatch,ebatch = batch, (eval_batch if eval_batch else batch)

        # A seeded generator makes the train/eval split identical on every rank
        # so the DistributedSampler shards a consistent partition.
        split_gen = None
        if dnn.dist.is_dist():
            split_gen = torch.Generator().manual_seed(ddp_split_seed)
        dses = dnn.data.train_eval_split(ds, train_ratio, generator=split_gen)

        # Under DDP shard each split with a DistributedSampler (mutually exclusive
        # with shuffle=); otherwise use plain shuffling as before.
        samplers = [None, None]
        if dnn.dist.is_dist():
            samplers = [DistributedSampler(dses[0], shuffle=True),
                        DistributedSampler(dses[1], shuffle=False)]
        dles = [DataLoader(one, batch_size=bb, shuffle=(sampler is None),
                           sampler=sampler, pin_memory=True)
                for one, bb, sampler in zip(dses, [tbatch, ebatch], samplers)]

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
            torch_seed = manual_seed,
            ddp_split_seed = ddp_split_seed,
            was_ddp = dnn.dist.is_dist(),
            # nested, not spread: a resolver reading this back must be able to
            # tell model config from run metadata, and spreading let a colliding
            # key ("name" being the plausible one) silently win.
            model_args = model_args,
        )
        run_history[this_run_number] = this_run

        epoch_history = history.get("epochs", dict())
        first_epoch_number = 0
        if epoch_history:
            first_epoch_number = max(epoch_history.keys()) + 1

        def saveit(path):
            if not path:
                return
            # Only rank 0 writes checkpoints under DDP (replicas are in sync, and
            # net here is the raw module so state-dict keys have no 'module.' prefix).
            if not dnn.dist.is_main():
                return
            dnn.io.save_checkpoint(path, net, opt, runs=run_history, epochs=epoch_history)

        for this_epoch_number in range(first_epoch_number, first_epoch_number + epochs):

            # Reshuffle the sharded training data differently each epoch.
            if samplers[0] is not None:
                samplers[0].set_epoch(this_epoch_number)

            train_loss = 0
            train_losses = []
            dt=0
            if ntrain:
                dt = time.time()
                train_loss, train_losses = trainer.epoch(dles[0])
                dt = time.time() - dt

            eval_loss = 0
            eval_losses = []
            edt = 0
            if neval:
                edt = time.time()
                eval_loss, eval_losses = trainer.evaluate(dles[1])
                edt = time.time() - edt

            this_epoch = dict(
                run=this_run_number,
                epoch=this_epoch_number,
                train_losses=train_losses,
                train_loss=train_loss,
                eval_losses=eval_losses,
                eval_loss=eval_loss)
            epoch_history[this_epoch_number] = this_epoch

            if dnn.dist.is_main():
                log.info(f'run: {this_run_number} epoch: {this_epoch_number} loss: {train_loss:.4e} [b={tbatch},n={ntrain}] eval: {eval_loss:.4e} [b={ebatch},n={neval}] {dt=:.3e} s {edt=:.3e} s')

            if checkpoint_save:
                if this_epoch_number % checkpoint_modulus == 0:
                    parms = dict(this_run, **this_epoch)
                    cpath = checkpoint_save.format(**parms)
                    saveit(cpath)
        saveit(save)
    finally:
        dnn.dist.cleanup()


def _parse_shape(val):
    '''
    Parse an input shape from a config value or CLI string.

    Accepts a real sequence, "[1, 3, 800, 1500]" or "1,3,800,1500".
    '''
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        return tuple(int(v) for v in val) or None
    text = str(val).strip()
    if not text:
        return None
    try:
        import ast
        got = ast.literal_eval(text)    # not eval(): this comes from a file
        if isinstance(got, (list, tuple)):
            return tuple(int(v) for v in got)
        return (int(got),)
    except (ValueError, SyntaxError):
        pass
    try:
        return tuple(int(v) for v in text.replace(',', ' ').split()) or None
    except ValueError:
        raise click.BadParameter(f'cannot read a shape from {val!r}, '
                                 'want e.g. "1,3,800,1500"')


def _parse_bool(val, default=False):
    '''
    Interpret a config-file boolean.

    Needed because anyconfig hands back the raw string and bool("false") is True.
    '''
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ('1', 'true', 'yes', 'on')


@cli.command('export-ts')
@click.option("-a", "--app", default=None, type=str,
              help="The application name (def: [export] app, else [train] app)")
@click.option("-l", "--load", default=None,
              help="File providing model weights: a training checkpoint or a bare "
              "state dict (def=None - export freshly initialized weights)")
@click.option("-o", "--output", default=None,
              help="Output TorchScript file (def: [export] output, else <app>.ts)")
@click.option("-d", "--device", default=None, type=str,
              help="Device to build and convert on (def=cpu).  Converting on CUDA "
              "bakes the device into the graph, so the result then needs a GPU.")
@click.option("-m", "--method", default=None,
              type=click.Choice(['script', 'trace']),
              help="Conversion method (def=script).  'script' needs no example input "
              "and stays shape-generic; 'trace' records one execution and is the "
              "fallback for a model that will not compile (e.g. xvunet).")
@click.option("--shape", default=None, type=str,
              help="Input shape, e.g. '1,3,800,1500'.  Required for --method trace, "
              "otherwise used to verify the export.  Def: [export] shape.")
@click.option("--sigmoid/--no-sigmoid", default=None,
              help="Append a sigmoid to the model output.  Needed for apps whose "
              "forward returns raw logits, e.g. dnnroi_custom.")
@anyconfig_file("wirecelldnn", section='export', defaults={})
@click.pass_context
def export_ts(ctx, app, load, output, device, method, shape, sigmoid, config):
    '''
    Export a model to a TorchScript (.ts) file.

    The model is constructed from the config exactly as "train" constructs it, so
    the [model] section supplies its arguments.  Weights are loaded from -l/--load
    if given, then the model is switched to eval() and converted.

    Settings come from an [export] section, with [train] as a fallback for "app".
    Anything given on the command line wins:

    \b
        [export]
        app = dnnroi_custom
        shape = [1, 3, 800, 1500]
        method = script
        sigmoid = true
        output = dnnroi_custom.ts
    '''
    import wirecell.dnn.apps

    # anyconfig_file hands back None, not {}, when there is no config file at all
    # (so `export-ts -a <app>` with no -c must still work).
    config = config or dict()
    xcfg = config.get('export', None) or dict()
    tcfg = config.get('train', None) or dict()

    # The decorator is given defaults={} because anyconfig_file dereferences
    # `defaults` unconditionally once its section exists (util/cli.py:329), so
    # section= without defaults= raises AttributeError.  An empty dict also keeps
    # its type coercion out of the way: values arrive as the raw config strings
    # and are converted below.  It skips the fill-in entirely when the section is
    # absent, so the fallbacks and defaults are resolved here either way.
    name = app or xcfg.get('app', None) or tcfg.get('app', None)
    if not name:
        raise click.BadArgumentUsage(
            'no app given; use -a/--app or set "app" in [export] or [train]')
    method = method or xcfg.get('method', None) or 'script'
    device = device or xcfg.get('device', None) or 'cpu'
    load = load or xcfg.get('load', None)
    output = output or xcfg.get('output', None) or f'{name}.ts'
    shape = _parse_shape(shape if shape is not None else xcfg.get('shape', None))
    # Only a --sigmoid/--no-sigmoid flag arrives as a real bool; a value taken
    # from [export] arrives as the raw string, and bool("false") is True.
    if not isinstance(sigmoid, bool):
        sigmoid = _parse_bool(sigmoid if sigmoid is not None
                              else xcfg.get('sigmoid', None), False)

    if method == 'trace' and shape is None:
        raise click.BadArgumentUsage(
            '--method trace needs an input shape; give --shape or set '
            '"shape" in [export]')

    app = getattr(wirecell.dnn.apps, name)
    net, _ = obj_with_config(app.Network, config, 'model')

    if load:
        if not Path(load).exists():
            raise click.FileError(load, 'model weights file does not exist')
        dnn.io.load_model_state(load, net)
        log.info(f'loaded weights from {load}')
    else:
        log.warning('no -l/--load given: exporting freshly initialized weights')

    nparam = sum(p.numel() for p in net.parameters())
    log.info(f'exporting {name} ({nparam/1e6:.2f} M parameters) via {method} '
             f'on {device}, sigmoid={sigmoid}')

    try:
        info = dnn.io.save_torchscript(net, output, shape=shape, method=method,
                                       device=device, sigmoid=sigmoid)
    except Exception as err:
        if method == 'script':
            # TorchScript errors can embed whole tensor reprs; keep the head.
            detail = ' '.join(str(err).split())
            if len(detail) > 300:
                detail = detail[:300] + ' ...'
            raise click.ClickException(
                f'scripting {name} failed: {detail}\n\n'
                'Scripting statically compiles the whole model, so it rejects '
                'things that only work in Python: indexing an nn.ModuleList with '
                'a runtime value, passing a function as an argument, or an '
                'attribute that changes type.  Note this is a compile-time '
                'limit, so eval() mode does not avoid it.\n'
                'Retry with "--method trace --shape ...", which records one '
                'execution instead and therefore sidesteps all of the above.'
            ) from err
        raise

    log.info(f'wrote {info["path"]}')
    if 'max_abs_diff' in info:
        log.info(f'  verified against the eager model: '
                 f'max|exported-eager| = {info["max_abs_diff"]:.3e}')
        log.info(f'  output {info["out_shape"]} range '
                 f'{info["out_min"]:.4f} .. {info["out_max"]:.4f}')
        # A range outside [0,1] with no sigmoid almost always means this app
        # returns logits, so the export is not what the consumer expects.
        if not info['sigmoid'] and (info['out_min'] < 0.0 or info['out_max'] > 1.0):
            log.warning('  output leaves [0,1] and sigmoid is off, so this model '
                        'appears to return logits.  If the consumer expects a '
                        'probability, re-export with --sigmoid.')
        if info['max_abs_diff'] > 0.0:
            log.warning('  exported model does not reproduce the eager model '
                        f'exactly (max|diff| = {info["max_abs_diff"]:.3e})')
    else:
        log.info('  no shape given, so the export was not verified; pass --shape '
                 'to check it against the eager model')
    if method == 'trace':
        log.info('  traced models fix the spatial dimensions they were exported '
                 'at (the batch dimension stays free)')


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

run_one_defaults = dict(device='cpu', name='dnnroi')
@cli.command('run_one')

@click.option("-d", "--device", default=None, type=str,
              help="The compute device")
@click.option("--debug-torch/--no-debug-torch", is_flag=True, default=False,
              help="Debug torch-level problems")
@click.option("-n", "--entry", default=0, help="Which entry to supply to DataLoader's __get_item__")
@click.option("-l", "--load", default=None,
              help="File name providing the initial model state dict (def=None - construct fresh)")
@click.option("-o", "--output", default=None,
              help="File name to output after training (def=None - results not saved)")
@click.option("-a", "--app", default=None, type=str,
              help="The application name")
@click.option('--manual-sigmoid/--no-manual-sigmoid', default=False, is_flag=True,
              help='Run output through sigmoid by hand')
@click.option('--rec-only', default=False, is_flag=True,
              help='Only run rec')
@click.option('--profile', default=None, type=str,
              help='Run profiling. Provide filename to store results. Default = None --> off')
@anyconfig_file("wirecelldnn", section='run_one', defaults=run_one_defaults)
@click.argument("files", type=str, nargs=-1)
def run_one(config, device, debug_torch, entry, load, output, app, manual_sigmoid, rec_only, profile, files):
    '''
    Run a reco & true pair through a saved model.
    '''
    # delay importing this monster
    from torch import save as torchsave, no_grad, sigmoid, cuda
    from torch.profiler import profile as do_profile, ProfilerActivity, record_function
    
    # import torch
    from torch.utils.data import DataLoader
    import wirecell.dnn.apps

    # if not files:               # args not processed by anyconfig_files
    #     try:
    #         files = config['train']['files']
    #     except KeyError:
    #         files = None
    if not files:
        raise click.BadArgumentUsage("no training files given")
    files = unglob(listify(files))
    log.info(f'training files: {files}')

    if device == 'gpu': device = 'cuda'

    name = app
    app = getattr(wirecell.dnn.apps, name)

    with no_grad():
        # As in train(): read first, so the seed-only keys can be dropped.  They
        # are pure waste here too -- the state dict below replaces every weight
        # they would have loaded -- and a cfg naming trunk checkpoints that have
        # since moved would otherwise fail before this model is ever run.
        ck = None
        if load:
            if not Path(load).exists():
                raise click.FileError(load, 'warning: DNN module load file does not exist')
            ck = dnn.io.load_checkpoint_raw(load)

        model_args = resolve_model_config(app.Network, config.get('model') or {}, ck)
        net = app.Network(**model_args)
        net = net.to(device)
        net.eval()

        if ck is not None:
            dnn.io.load_model_state_from(ck, net, path=load)
            log.info(f'loaded model state from {load}')
            ck = None           # see train(): load_state_dict copied already

        ds = app.Dataset(files, config=config.get("run_one_dataset", None), rec_only=rec_only)
        if len(ds) == 0:
            raise click.BadArgumentUsage(f'no samples from {len(files)} files')
        feat, labels = ds.__getitem__(entry)
        # print(feat.shape)
        # print(labels.shape)
        x = feat.to(device).unsqueeze(0)
        # y = net(feat.to(device).unsqueeze(0)).squeeze(0)

        #Set up profiling
        activities = [ProfilerActivity.CPU]
        sort_by = "cpu_time_total"
        if cuda.is_available() and ('cuda' == device):
            activities += [ProfilerActivity.CUDA]
            sort_by = 'cuda_time_total'

        def call_net(net, x):
            y = net(x).squeeze(0)
            if manual_sigmoid:
                y = sigmoid(y)
            return y
            
        if profile:
            with do_profile(activities=[ProfilerActivity.CPU]) as prof:
                with record_function("model_inference"):
                    y = call_net(net, x)
            print(prof.key_averages().table(sort_by=sort_by, row_limit=10))
            prof.export_chrome_trace(profile)
        else:
            y = call_net(net, x)

    if output:
        outdict = {
            'feat':feat,
            'y':y
        }
        if not rec_only: outdict['labels'] = labels
        torchsave(outdict, output)

# run_one_defaults = dict(device='cpu', name='dnnroi')
@cli.command('viztrain')

@click.option("-o", "--output", default=None,
              help="File name to output (does not show the image before saving)")
@click.option('--mean-train', is_flag=True, help='Average training loss over epoch -- default: display loss for each input/batch')
@click.option('--eval-only', is_flag=True, help='Just display eval loss with no averaging. Training loss is not displayed')
@click.option('--no-dots', is_flag=True, help='Turn off drawing dots for eval loss when drawing per-sample training loss')
@click.option('--logy', is_flag=True, help='Log-scale for y-axis')
#Another option: batch ratio scaling?
@click.argument("checkpoint", type=str)
def viztrain(output, checkpoint, mean_train, eval_only, no_dots, logy):
    '''Visualize training curves from a checkpoint file'''
    import matplotlib.pyplot as plt
    from torch import load, device
    import numpy as np

    f = load(checkpoint, map_location=device('cpu'))
    epochs = f['epochs'].keys()
    #Better be sure these exist
    train_losses = np.array([f['epochs'][k]['train_losses'] for k in epochs])
    eval_losses = np.array([f['epochs'][k]['eval_losses'] for k in epochs])
    eval_xs = (np.arange(len(eval_losses)) +  1)*len(train_losses[0])
    
    
    if eval_only:
        xlabel = 'Epoch'
        plt.plot(eval_losses.flatten(), label='Eval Loss')
    else:
        if mean_train:
            plt.plot(np.mean(train_losses, axis=1).flatten(), label='Training Loss')
            plt.plot(np.mean(eval_losses, axis=1).flatten(), label='Eval Loss')
            xlabel = 'Epoch'
        else:
            xlabel = 'Sample'
            plt.plot(train_losses.flatten(), label='Training Loss', zorder=1)
            plt.plot(eval_xs, np.mean(eval_losses, axis=1).flatten(), label='Eval Loss', zorder=1)
            if not no_dots:
                plt.scatter(eval_xs, np.mean(eval_losses, axis=1).flatten(), c='tab:orange', label='Eval Loss', zorder=2)
    
    plt.legend(fontsize=12)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    if logy: plt.yscale('log')
    
    if output: plt.savefig(output)
    else: plt.show()



def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
