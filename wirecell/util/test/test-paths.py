#!/usr/bin/env pytest

import pytest
from warnings import warn
import os
from pathlib import Path
from wirecell.util.paths import listify, resolve

def test_listify():
    assert listify() == ()
    assert listify(None) == ()
    assert listify("aa") == ("aa",)
    assert listify("aa:bb") == ("aa","bb")
    assert listify(["aa:bb","cc"]) == ("aa","bb","cc")
    assert listify(["aa:bb","cc",["dd"]]) == ("aa","bb","cc","dd")
    assert listify(["aa:bb","cc"],["dd"]) == ("aa","bb","cc","dd")
    assert listify(["aa:bb"],["cc","dd"]) == ("aa","bb","cc","dd")

def test_resolve():
    me = Path(__file__)
    assert resolve(me) == me
    assert resolve(me.absolute()) == me
    assert resolve(me.name, me.parent) == me

def test_resolve_no_wirecell_path():

    wcp = os.environ.get('WIRECELL_PATH', "")
    os.environ['WIRECELL'] = ""
    with pytest.raises(FileNotFoundError):
        resolve("wirecell.jsonnet")
    os.environ['WIRECELL'] = wcp

def test_resolve_with_wirecell_path():
    wcp = os.environ.get('WIRECELL_PATH', "")
    if not wcp:
        pytest.skip('Skipping test of resolve due to undefined WIRECELL_PATH')
    resolve("wirecell.jsonnet")


