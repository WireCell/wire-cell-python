#!/usr/bin/env python
'''
Tests for the "wcpy pgraph importify" helpers: the lexical (static) import graph
and the eval (lazily-forced) import graph.
'''
import os
import pytest

from wirecell.pgraph.__main__ import (
    scan_imports, resolve_import, lexical_graph, eval_graph,
)


def _write(d, name, text):
    path = os.path.join(str(d), name)
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(text)
    return path


@pytest.fixture
def tree(tmp_path):
    '''A small import tree.

    root imports "used" (returned, so forced) and "unused" (bound but never
    referenced, so lazy).  "unused" itself imports a non-existent file, which
    must NOT error unless "unused" is forced.
    '''
    root = _write(tmp_path, "root.jsonnet",
                  'local used = import "used.jsonnet";\n'
                  'local unused = import "unused.jsonnet";\n'
                  'used\n')
    _write(tmp_path, "used.jsonnet", '{ a: 1 }\n')
    _write(tmp_path, "unused.jsonnet", 'import "missing.jsonnet"\n')
    return tmp_path, root


def test_scan_imports(tree):
    _d, root = tree
    imps = scan_imports(root)
    assert ("import", "used.jsonnet") in imps
    assert ("import", "unused.jsonnet") in imps


def test_scan_imports_ignores_comments(tmp_path):
    root = _write(tmp_path, "c.jsonnet",
                  '// local x = import "commented.jsonnet";\n'
                  '# import "hashed.jsonnet"\n'
                  'local real = import "real.jsonnet";\n'
                  'real\n')
    imps = dict((rel, kind) for kind, rel in scan_imports(root))
    assert "real.jsonnet" in imps
    assert "commented.jsonnet" not in imps
    assert "hashed.jsonnet" not in imps


def test_resolve_import_importer_dir_first(tree):
    d, root = tree
    got = resolve_import(root, "used.jsonnet", [])
    assert got == os.path.join(str(d), "used.jsonnet")
    assert resolve_import(root, "nope.jsonnet", []) is None


def test_lexical_is_complete_superset(tree):
    d, root = tree
    edges, unresolved = lexical_graph(root, [])
    used = os.path.join(str(d), "used.jsonnet")
    unused = os.path.join(str(d), "unused.jsonnet")
    targets = {dst for _s, dst, _k in edges}
    # Lexical sees BOTH imports, even the never-forced one.
    assert used in targets
    assert unused in targets
    # And the dangling import under the unused branch is reported, not fatal.
    assert any(rel == "missing.jsonnet" for _s, rel, _k in unresolved)


def test_eval_excludes_unforced_branch(tree):
    from wirecell.util import jsio
    try:
        jsio.jsonnet_module()
    except ImportError:
        pytest.skip("no jsonnet binding available")
    d, root = tree
    edges, nodes = eval_graph(root, [], {})
    used = os.path.join(str(d), "used.jsonnet")
    unused = os.path.join(str(d), "unused.jsonnet")
    # Only the forced import is evaluated; the lazy branch (and its dangling
    # import) never appear, so evaluation succeeds.
    assert used in nodes
    assert unused not in nodes
    targets = {dst for _s, dst, _k in edges}
    assert used in targets
    assert unused not in targets
