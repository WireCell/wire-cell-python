
def train_eval_split(ds, train_ratio, **dlkwds):
    '''
    Return pair (train,eval) of datasets formed by a random split of dataset

    The "train" will have train_ratio of the total number of samples in the dataset.

    The train ratio is clamped to be in [0.0, 1.0].

    Note, train dataset will be empty if ratio is 0.0, etc eval for 1.0.

    '''
    # delay loading in the monster
    from torch.utils.data import random_split

    # clamp
    train_ratio = sorted((0.0, train_ratio, 1.0))[1]

    nds = len(ds)
    ntrain = int(nds*train_ratio)
    neval = nds - ntrain
    return random_split(ds, [ntrain, neval])

    # train_dl = DataLoader(train_ds, **dlkwds)
    # eval_dl = DataLoader(train_ds, **dlkwds)
    # return train_dl, eval_dl

