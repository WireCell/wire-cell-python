#!/usr/bin/env python
import wirecell
from wirecell.util import tdm

def sane(n):
    assert 'metadata' not in n
    assert 'metadata' not in n.md    
    assert 'array' not in n
    assert 'array' not in n.md    

def test_tree_empty():
    t = tdm.Tree()
    sane(t)
    assert isinstance(t.md, dict)
    assert isinstance(t.metadata, dict)
    assert t.array is None
    sane(t)

def test_tree_deep():
    path = "path/to/some/deep/leaf"

    t = tdm.Tree()
    l = t(path)
    assert isinstance(l.metadata, dict)
    sane(t)

    path = path.split('/')
    for n in range(len(path)+1):
        p = list(path)
        p.insert(n, '/')
        ll = t(p)
        assert id(l) == id(ll)
        sane(t)

    c = tdm.Tree({"name":"child"})
    print(c)
    print(c.md)
    sane(c)

    t.insert("path/to/child", c)
    assert id(t("path/to/child")) == id(c)

    got = t.visit(lambda n,c: ("/".join(c),n.md), with_context=True)
    d = {k:v for k,v in got}
    assert d["path/to/child"]["name"] == "child"
    sane(t)
    print(t)
    print(t.md)

def test_tree_metadata():
    t = tdm.Tree({"a":1},b=2)
    assert len(t) == 0
    assert t.array is None

    assert t.md['a'] == 1
    assert t.md['b'] == 2
    assert t.a == 1
    assert t.b == 2
    t.a = 3
    assert t.a == 3
    sane(t)

def test_tree_array():
    t = tdm.Tree(array=object())
    assert isinstance(t, object)
    sane(t)    
