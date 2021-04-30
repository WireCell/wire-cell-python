#!/usr/bin/env pytest

from wirecell.sigproc.garfield import parse_filename

def test_parse_filename():
    fnames = [
        "/home/bv/work/pcbro/fields/dune_4.71/0.0_U.dat",
        "dune_4.71/0.0_U.dat",
        "0.0_U.dat"
    ]
    for fname in fnames:
        got = parse_filename(fname)
        print(f'{got} ({fname})')
        assert got['filename'] == fname
        assert got['impact'] == 0.0
        assert got['plane'] == 'u'
        
