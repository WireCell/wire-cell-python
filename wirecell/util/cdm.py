#!/usr/bin/env python3
'''
Things for cluster data model
'''

def looks_like(fp):
    for key in fp.keys():
        if 'cluster_' in key: return True
    return False

def dumps(fp):
    lines = list()
    for key in fp:
        o = fp[key]
        
        if isinstance(o, dict):
            keys = ', '.join(o.keys())
            lines.append(f'{key:16s}\t{keys}')
            continue
        lines.append(f'{key:16s}\t{o.dtype}\t{o.shape}')
    return '\n'.join(lines)
    
