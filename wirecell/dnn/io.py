#!/usr/bin/env python
'''
File I/O


https://pytorch.org/tutorials/beginner/saving_loading_models.html
'''

import torch

def save_checkpoint(path, model, optimizer, **kwds):
    '''
    Save a checkpoint to file at path.

    Checkpoint consists of model and optimizer state dicts and any additional
    attributes supplied as kwds.
    '''
    kwds.update(model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict())
    torch.save(kwds, path)


def load_checkpoint_raw(path):
    return torch.load(path, weights_only=True)
    
def load_only_model(path, model, strict=True):
    cp = load_checkpoint_raw(path)
    model.load_state_dict(cp.pop("model_state_dict"), strict=strict)
    return cp
def load_checkpoint(path, model, optimizer):
    '''
    Load a checkpoint.

    The model and optimizer state dicts are updated and a dict of any additional
    parameters is returned.
    '''
    cp = load_checkpoint_raw(path)
    model.load_state_dict(cp.pop("model_state_dict"))
    optimizer.load_state_dict(cp.pop("optimizer_state_dict"))
    return cp