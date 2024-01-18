#!/usr/bin/env pytest

from wirecell.util.peaks import *
from wirecell.util.codec import JsonEncoder

def roundtrip_dataclass(DCType):
    t = DCType()
    d = t.to_dict()
    print(d)
    j = json.dumps(d, cls=JsonEncoder)
    d2 = json.loads(j)
    print(d2)
    assert d == d2
    t2 = DCType.from_dict(d2)


def test_dataclasses():
    roundtrip_dataclass(Peak1d)
    
