#!/usr/bin/env python
'''
Give some structure to Bee's json.

The data representation and loading is based on the Bee file format
documentation at:

  https://bnlif.github.io/wire-cell-docs/viz/uploads/

'''
import json
import numpy
from scipy.spatial import KDTree
from collections import defaultdict
from pathlib import Path
from wirecell.util import ario
import logging
log = logging.getLogger("wirecell.bee")

class Cluster:
    '''
    Information of points with a common cluster ID.
    '''
    def __init__(self, ident, points):
        # cluster_id, real_cluster_id, q
        self.ident = ident
        self.points = points
        self.kd = KDTree(self.points)
        # fixme/todo: add PCAs, cop, coq


class Grouping:
    '''
    A grouping of clusters from the same "algorithm".

    Clusters are accessed by their cluster ID.

    A grouping must be atomic in a data file.  Access children via .clusters attribute.
    '''
    def __init__(self, index, algname, content):
        self.index = index
        self.algname = algname

        self.name = content.get("type","")
        self.rse = (content.get("runNo",0), content.get("subRunNo",0), content.get("eventNo",0))
        self.geom = content.get("geom","")

        self.points = numpy.array([content[axis] for axis in "xyz"]).T

        cpts = defaultdict(list)
        for ind, cid in enumerate(content["cluster_id"]):
            cpts[cid].append(self.points[ind])

        self.clusters = defaultdict(Cluster)
        for cid, pts in cpts.items():
            self.clusters[cid] = Cluster(cid, numpy.array(pts))

Ensemble = dict # index a grouping by "algname"
class Series(dict):

    def add_group(self, grp):
        if grp.index not in self:
            self[grp.index] = Ensemble()
        self[grp.index][grp.algname] = grp

    def add_ensemble(self, ens):
        for grp in ens.values():
            self.add_group(grp)

    def add_series(self, ser):
        for ens in ser.values():
            self.add_ensemble(ens)


def load_json(json_file):
    '''
    Load a single Grouping from a JSON file.
    '''
    index, algname = parse_pathname(json_file)
    content = json.loads(open(json_file).read())
    return Grouping(index, algname, content)

    
def parse_pathname(path):
    path = Path(path)
    parts = path.stem.split('-', 1)
    if len(parts) != 2:
        raise ValueError(f'weird path: "{path}", got {parts}')
    return tuple(parts)

def load_zip(zip_file):
    '''
    Load a zip file, return the series.
    '''
    series = Series()
    for pathname, content in ario.load(zip_file).items():
        index, algname = parse_pathname(pathname)
        grp = Grouping(index, algname, content)
        series.add_group(grp)
    return series
    

def load(sources):
    '''
    Load zip or json files, return a Series object.

    The source may be a single or a sequence of files paths as string or
    pathlib.Path objects.  
    '''
    # pluralize
    if isinstance(sources, (str, Path)):
        sources = [sources]

    series = Series()
    for source in sources:
        if source.endswith(".json"):
            grp = load_json(source)
            series.add_group(grp)
            continue
        if source.endswith(".zip"):
            series.add_series(load_zip(source))
            continue
        log.warn(f'unsupported source: "{source}"')
        continue
    return series
