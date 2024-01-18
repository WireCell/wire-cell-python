#!/usr/bin/env pytest

import json
import dataclasses
from wirecell.gen import linegen 
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


def test_trackconfig_roundtrip():
    roundtrip_dataclass(linegen.TrackConfig)

def test_metadata_roundtrip():
    roundtrip_dataclass(linegen.TrackMetadata)

