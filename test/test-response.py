#!/usr/bin/env pytest
'''
Test wirecell.sigproc.response
'''
import json
import pytest
import numpy
from itertools import product
from wirecell.sigproc.response import persist, schema


def assert_equal(fr1, fr2):
    assert isinstance(fr1, schema.FieldResponse)
    assert isinstance(fr2, schema.FieldResponse)

    
    assert fr1.origin == fr2.origin
    assert fr1.tstart == fr2.tstart
    assert fr1.period == fr2.period
    assert fr1.speed == fr2.speed

    assert len(fr1.planes) == len(fr2.planes)
    for pl1,pl2 in zip(fr1.planes, fr2.planes):
        print (f'PlaneRespones: {type(pl1)} {type(pl2)}')
        assert isinstance(pl1, schema.PlaneResponse)
        assert isinstance(pl2, schema.PlaneResponse)

        assert pl1.planeid == pl2.planeid
        assert pl1.location == pl2.location
        assert pl1.pitch == pl2.pitch

        assert len(pl1.paths) == len(pl2.paths)
        for pa1,pa2 in zip(pl1.paths, pl2.paths):
            assert isinstance(pa1, schema.PathResponse)
            assert isinstance(pa2, schema.PathResponse)

            assert pa1.pitchpos == pa2.pitchpos
            assert pa1.wirepos == pa2.wirepos
            assert numpy.all(pa1.current == pa2.current)
            


def check_one(fr, name, tmp_path, ext):
    assert isinstance(fr.planes[0], schema.PlaneResponse)

    print("testing in memory round trip")
    pod = persist.schema2pod(fr)
    assert 'FieldResponse' in pod
    pfr = pod['FieldResponse']
    print('FieldResponse keys:', pfr.keys())
    pr0 = pfr['planes'][0]
    print('PR0 keys', pr0.keys())
    assert 'PlaneResponse' in pr0

    fr2 = persist.pod2schema(pod)
    assert_equal(fr, fr2)       # in memory round trip


    fname = tmp_path / f'response-{name}.{ext}'
    persist.dump(fname, fr)

    fr2 = persist.load(fname)

    assert_equal(fr, fr2)       # round trip via file

def round_trip(name, tmp_path, ext):
    fr = persist.load(name)
    if not isinstance(fr, list):
        fr = [fr]
    for ind, one in enumerate(fr):
        check_one(one, f'{name}{ind}', tmp_path, ext)

# Note, converting from schema to array form is lossy due to each row of the
# array representing a bin-centered average of two paths.
@pytest.mark.parametrize('det,ext', product(('pdsp','uboone'), ('json',)))
def test_round_trip(tmp_path, det, ext):
    round_trip(det, tmp_path, ext)

def test_types():
    fr = schema.FieldResponse()
    assert not isinstance(fr, tuple)
    assert isinstance(fr, schema.FieldResponse)
    frd = fr.to_dict()
    assert isinstance(frd, dict)
