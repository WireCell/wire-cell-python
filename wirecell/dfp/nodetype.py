import json
from collections import namedtuple

NodeType = namedtuple("NodeType","type category input_types output_types concurrency")
def make(type="", category=-1, input_types=None, output_types=None, concurrency=1):
    '''Return a representation of information about a concrete
    WireCell::INode class with some defaults.    '''
    return NodeType(type, category, input_types or list(), output_types or list(), concurrency)
    
def loads(jsonstr):
    '''
    Load a JSON string, such as produced by WireCellApps::NodeDumper, into dict of NodeType keyed by type.
    '''
    dat = json.loads(jsonstr)
    ret = dict()
    for d in dat:
        nt = make(**d)
        ret[nt.type] = nt
    return ret


def to_dots(desc):
    '''
    Return a dot string representation of the description of node types.
    '''
    
