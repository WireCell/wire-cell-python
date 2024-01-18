import json
import numpy
import dataclasses

def to_pod(v):
    '''
    Try hard to return v as POD
    '''
    if isinstance(v, numpy.ndarray):
        return v.tolist()
    if isinstance(v, slice):
        return [v.start, v.stop, v.step]
    if dataclasses.is_dataclass(v):
        return dataclasses.asdict(v, dict_factory = dict_factory)
    return v

def dict_factory(kv):
    '''
    Try hard to convert dataclass key/values to POD.

    Sutable for calls like:

        ddict = dataclasses.asdict(dclass, dict_factory=dict_factory)
    '''
    return {k:to_pod(v) for k,v in kv}

@classmethod
def from_dict(cls, obj = {}):
    '''
    Return instance of dataclass from dict-like POD.
    '''
    dat = {f.name: f.type(obj.get(f.name, f.default))
           for f in dataclasses.fields(cls)}
    return cls(**dat)

def to_dict(self):
    '''
    Try hard to return a dataclass as a dict of POD.
    '''
    return dataclasses.asdict(self, dict_factory=dict_factory)

def dataclass_dictify(cls):
    '''
    Decorate a dataclass to add from_dict(cls) and to_dict(self) methods.
    '''
    cls.from_dict = from_dict
    cls.to_dict = to_dict
    return cls

class JsonEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        if isinstance(obj, numpy.floating):
            return float(obj)
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()        
        if isinstance(obj, slice):
            return (obj.start, obj.stop, obj.step)
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return super().default(obj)

    
def json_dumps(obj, **kwds):
    return json.dumps(obj, cls=JsonEncoder, **kwds)
