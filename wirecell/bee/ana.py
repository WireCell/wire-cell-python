#!/usr/bin/env python
'''
Functions to analyze bee.data.
'''

levels = ["point","shape","cluster","grouping","ensemble"]
def level_index(lvl):
    if isinstance(lvl, int):
        return lvl
    return levels.index(lvl)


class Summary:

    def __init__(self, ser, level='cluster'):
        self._ser = ser
        self._pad = '  '
        self._level = level_index(level)

    def __str__(self):
        return self.series(self._ser)
        
    def point(self, pt, tab=0):
        gap = self._pad*tab
        return f'{gap}pt:{pt}'

    def cluster(self, cls, tab=0):
        gap = self._pad*tab
        gapp = self._pad*(tab+1)

        # fixme/todo: add PCAs, cop, coq
        lines = [f'{gap}Cluster: id:{cls.ident} npts:{len(cls.points)}']
        if self._level <= level_index("shape"):
            for ind, (vec, val) in enumerate(zip(*cls.pca_eigen)):
                lines.append(f'{gapp}pca {ind}: {val} {vec}')
        if self._level <= level_index("point"):
            for pt in cls.points:
                lines.append(self.point(pt, tab+2))
            
        return '\n'.join(lines)


    def grouping(self, grp, tab=0):
        gap = self._pad*tab
        parts = [f'{gap}Grouping ind:{grp.index} nclusters:{len(grp.clusters)} npoints:{len(grp.points)} rse:{grp.rse} alg:"{grp.algname}" type:"{grp.name}"']
        if self._level < level_index("grouping"):
            for cls in grp.clusters.values():
                parts.append(self.cluster(cls, tab+1))
        return '\n'.join(parts)


    def ensemble(self, ens, ind='', tab=0):
        gap = self._pad*tab
        parts = [f'{gap}Ensemble {ind}']
        if self._level < level_index("ensemble"):
            for grp in ens.values():
                parts.append(self.grouping(grp, tab+1))
        return '\n'.join(parts)


    def series(self, ser):
        '''
        Return a text summary of the series.
        '''
        parts = list()
        for ind, ens in ser.items():
            parts.append(self.ensemble(ens, ind, 1))
        return '\n'.join(parts)

        
    
