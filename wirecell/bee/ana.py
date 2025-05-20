#!/usr/bin/env python
'''
Functions to analyze bee.data.
'''

def summarize_cluster(cls, tab=0, pad='  '):
    gap = pad*tab
    gapp = pad*(tab+1)

    # fixme/todo: add PCAs, cop, coq
    lines = [f'{gap}Cluster: {cls.ident} {len(cls.points)}']
    for ind, (vec, val) in enumerate(zip(*cls.pca_eigen)):
        lines.append(f'{gapp}pca {ind}: {val} {vec}')
    return '\n'.join(lines)


def summarize_grouping(grp, tab=0, pad='  '):
    gap = pad*tab
    parts = [f'{gap}Grouping {grp.index} "{grp.algname}" "{grp.name}" {grp.rse} {len(grp.clusters)} {len(grp.points)}']
    for cls in grp.clusters.values():
        parts.append(summarize_cluster(cls, tab+1, pad))
    return '\n'.join(parts)


def summarize_ensemble(ens, ind='', tab=0, pad='  '):
    gap = pad*tab
    parts = [f'{gap}Ensemble {ind}']
    for grp in ens.values():
        parts.append(summarize_grouping(grp, tab+1, pad))
    return '\n'.join(parts)


def summarize_series(ser):
    '''
    Return a text summary of the series.
    '''
    parts = list()
    for ind, ens in ser.items():
        parts.append(summarize_ensemble(ens, ind, 1))
    return '\n'.join(parts)
