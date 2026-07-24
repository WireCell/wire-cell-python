#!/usr/bin/env python
'''
Single-node multi-GPU support via PyTorch DistributedDataParallel (DDP).

This module is a thin wrapper around ``torch.distributed`` that is driven by
the environment variables set by ``torchrun`` (``LOCAL_RANK``, ``RANK``,
``WORLD_SIZE``).  When those are absent, or indicate a single process, every
function here is a no-op so the plain ``wcpy dnn train`` path is unchanged.

Launch multi-GPU training with, e.g.::

    torchrun --standalone --nproc_per_node=4 -m wirecell.dnn train \\
        -a dnnroi -d cuda -b 8 -e 50 <files...>

``torch`` is imported lazily because it is an optional extra of this package.
'''

import os

# Module-level flag set by setup().  None means setup() has not run yet.
_is_dist = False


def _world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))


def setup():
    '''
    Initialize the distributed process group if launched under torchrun with
    more than one process.  Returns True if DDP is active, False otherwise.

    Safe to call unconditionally; a no-op for the single-process case.
    '''
    global _is_dist

    if _world_size() <= 1:
        _is_dist = False
        return False

    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(get_local_rank())
    _is_dist = True
    return True


def is_dist():
    'True if DDP has been initialized for this process.'
    return _is_dist


def get_local_rank():
    'GPU index this process owns on the local node (0 when not distributed).'
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_rank():
    'Global process index across all nodes (0 when not distributed).'
    return int(os.environ.get("RANK", "0"))


def get_world_size():
    'Total number of processes (1 when not distributed).'
    return _world_size()


def is_main():
    'True on the rank-0 process (always True when not distributed).'
    return get_rank() == 0


def all_reduce_sum(tensor):
    '''
    In-place sum-reduce a tensor across all ranks.  A no-op when not
    distributed.  Returns the tensor for convenience.
    '''
    if _is_dist:
        import torch.distributed as dist
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def cleanup():
    'Tear down the process group if it was initialized.  Safe to call always.'
    global _is_dist
    if _is_dist:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    _is_dist = False
