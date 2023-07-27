#!/usr/bin/env python3
'''
Code to work with WCT tensor data model files.
'''

from collections import namedtuple, defaultdict

def looks_like(fp):
    for key in fp.keys():
        if 'tensor_' in key: return True
        if 'tensorset_' in key: return True
    return False

class Tree(dict):
    '''Simple model of HDF5 style data structure.

    A tree is a dict-like object with tree-like values and with
    attributes:

    The dict interface is used to represent the tree.

    The fixed attributes are provided:

    .array :: an array object or None

    When .array is None the Tree (node) represents a "group" and if
    not None a "dataset".

    .metadata :: a dict    
    .md :: a shorthand alias
    
    .<metadata-key>

    Other requests for attributes not otherwise provided by Tree or
    dict will attempt to map to a .metadata key.  A .metadata key that
    is identical to a Tree or dict attribute must be accessed via the
    .metadata dict and not via a Tree attribute.

    '''

    def __init__(self, metadata=None, array=None, **md):
        '''
        Create a tree (node) with optional array and metadata dictionary.

        keyword arguments will also be added as additional metadata.
        '''
        self.__dict__['metadata'] = metadata or dict()
        self.metadata.update(md)
        self.__dict__['array'] = array

    @property
    def md(self):
        return self.metadata

    def __missing__(self, key):
        'Spawn a new child'
        value = self[key] = type(self)()
        return value

    def _path(self, path):
        if isinstance(path, str):
            path = path.split("/")
        path = [p.strip() for p in path]
        return [p for p in path if p and p != "/"]

    def __call__(self, path):
        '''
        Return a descendant tree at relative path from self. 

        The path is a list or "/"-separated string.

        Any missing descendants of this tree will be constructed.
        '''
        path = self._path(path)
        node = self
        for name in path:
            node = node[name]
        return node
            
    def visit(self, visitor, keys=lambda n: n.keys(), with_context=False):
        '''Recurse through tree calling visitor.

        Visitor is called as:

            visitor(node)          with_context is False
        or
            visitor(node, path)    with_context is True

        where "path" is a list giving the tree path to the node.

        It is called first on this node and then on each descendant
        node.  Its return values are collected in a list and returned.

        The keys function takes a node and should return a list of key
        names of children nodes on which to recurse.

        '''
        def v(node, context):
            if with_context:
                ret = [visitor(node, context)]
            else:
                ret = [visitor(node)]
            for key in keys(node):
                child = node[key]            
                ret += v(child, context + [key])
            return ret
        return v(self, [])

    def insert(self, path, node):
        '''
        Insert Tree node at path
        '''
        path = self._path(path)
        parent = self
        while len(path) > 1:
            parent = parent[path.pop(0)]
        child = path.pop(0)
        parent[child] = node

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        return self.metadata[key]

    def __setattr__(self, key, val):
        if key in self.__dict__:
            self.__dict__[key] = val
            return
        self.metadata[key] = val


def dumps(fp):
    '''
    Return dict summarizing a tensor data model file.

    >>> aa = ario.load("filename.npz")
    >>> s = dumps(aa)
    >>> print(json.dumpa(s))
    '''
    lines = []
    for key in fp:
        o = fp[key]
        
        if isinstance(o, dict):
            dpath = o.get('datapath',"")
            dtype = o.get('datatype', "")
            lines.append(f'{key:20s} {dtype:16s} {dpath}')
            continue
        dtype = str(o.dtype)        
        lines.append(f'{key:20s} {dtype:16s} {o.shape}')
    return '\n'.join(lines)


def load(af, ident=0):
    '''Load an ario-like file in tensor-data-model to a list of Tree.

    Each Tree in the list represents one tensorset with its tensors.

    If a tensorset provides a "datapath" metadata item the Tree (node)
    representing the tensorset will be found in the Tree at that path.
    Otherwise, the tensorset is represented by the top Tree (node)
    found as a list element.

    Any tensors will be placed in Tree at a location given by their
    (required) "datapath" metadata item.

    '''
    # ordered list of dataset idents
    idents = list()
    # top tree for each dataset keyed by str(ident)
    tops = defaultdict(Tree)

    def get_top(ident):
        if ident not in idents:
            idents.append(ident)
        return tops[ident]

    # tree for each tensor keyed by (str(ident),str(index)) tuple
    tens = defaultdict(lambda: defaultdict(Tree))
    def get_ten(ident, index):
        return tens[ident][index]

    for key in af:

        if key.startswith('tensorset_'):
            md = af[key]
            _,ident,_ = key.split("_")
            top = get_top(ident)
            if 'datapath' in md:
                top(md['datapath']).metadata = md
            else:
                top.metadata = md
            continue

        if key.startswith('tensor_'):
            _,ident,index,kind = key.split("_")
            ten = get_ten(ident,index)
            if kind == 'metadata':
                ten.metadata = af[key]
                continue
            if kind == "array":
                ten.array = af[key]
                continue
        
    ret = list()
    for ident in idents:
        top = get_top(ident)
        for index, ten in sorted(tens[ident].items()):
            dpath = ten.md['datapath']
            top.insert(dpath, ten)
        ret.append(top)
    return ret

def tohdf(trees, hd):
    '''
    Fill HDF5 file like object hd with Trees.
    '''
    import h5py

    if isinstance(trees, Tree):
        trees = [trees]

    def construct(node, path):
        dpath = '/'.join(path)
        dpath = "/"+dpath
        if node.array is not None:
            hnode = hd.create_dataset(dpath, data=node.array)
        else:
            try:
                hnode = hd.create_group(dpath)
            except ValueError:
                hnode = hd[dpath]

        for k,v in node.md.items():
            if k in ("datapath","arrays", "tensors"):
                continue

            hnode.attrs[k] = v
        return hnode

    def softlink(node, path):
        dpath = '/'.join(path)
        dpath = "/"+dpath
        for k,v in node.md.items():
            if k in ("arrays", "tensors"):
                for n,p in v.items():
                    dst = h5py.SoftLink(f'/{p}')
                    src = f'{dpath}/{n}'
                    hd[src] = dst

    for tree in trees:
        tree.visit(construct, with_context=True)
        tree.visit(softlink, with_context=True)
        


    # mds = dict()
    # arrs = dict()

    # for key in af:
    #     thing,ident = key.split('_',1)

    #     if thing == 'tensorset':
    #         md = af[key]
    #         dpath = md.pop('datapath', None)
    #         if dpath is None:
    #             continue
    #         grp = hd.create_group(dpath)
    #         for k,v in md.items():
    #             grp.attrs[k] = v
    #         continue

    #     thing,ident,index,kind = key.split('_')
    #     ii = (int(ident), int(index))

    #     if kind == 'metadata':
    #         mds[ii] = af[key]
    #         continue

    #     if kind == 'array':
    #         arrs[ii] = af[key]
    #         continue

    # for ii,arr in arrs.items():
    #     md = mds.pop(ii, {})
    #     dpath = md.pop('datapath')
    #     ds = hd.create_dataset(dpath, data=af[key])
    #     for k,v in md.items():
    #         ds.attrs[k] = v

    # for md in mds.values():
    #     dpath = md.pop('datapath')

    #     try:
    #         grp = hd.create_group(dpath)
    #     except ValueError:
    #         grp = hd[dpath]

    #     dtype = md['datatype']
    #     if dtype == 'pcdataset':
    #         arrays = md.pop('arrays')
    #         for aname, apath in arrays.items():
    #             grp[aname] = h5py.SoftLink(apath)

    #     # whatever is left
    #     for k,v in md.items():
    #         grp.attrs[k] = v
            
        
