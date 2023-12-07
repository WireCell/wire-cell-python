#!/usr/bin/env pytest
import pytest
from wirecell.util import detectors

def test_environment():
    p = detectors.registry_path()
    assert p
    assert p.exists()

def test_resolve():
    print()
    for name in ["pdsp","uboone"]:
        w = detectors.resolve(name, "wires")
        print (f'{name} wires: {w}')
        assert w
        assert w.exists()
        
        
def test_resolve_list():
    # MB has multiple field files
    f = detectors.resolve("uboone", "fields")
    assert isinstance(f, list)

def test_resolve_missing():
    # icarus has no normal noise file
    with pytest.raises(KeyError):
        detectors.resolve("icarus", "noise")
