#!/usr/bin/python3
'''
Fixme: make this into a proper click main
'''
import os
import re
import sys
import json
from collections import defaultdict
import subprocess

import click

from wirecell import units
from wirecell.util import jsio
from wirecell.util.cli import jsonnet_loader
from wirecell.util.cli import context, log

@context("pgraph")
def cli(ctx):
    '''
    Wire Cell Signal Processing Features
    '''
    pass

class Node (object):
    def __init__(self, tn, params=True, **attrs):
        log.debug(f'Node("{tn}") {params=} {attrs}')
        if not attrs:
            log.debug ("Node(%s) with no attributes"%tn)

        self.tn = tn
        self._params = params
        tn = tn.split(":")
        self.type = tn[0]
        try:
            self.name = tn[1]
        except IndexError:
            self.name = ""
        self.ports = defaultdict(set);
        pnode = attrs.pop('_pnode', {})
        for n in range(pnode.get("nin", 0)):
            self.add_port('head', n);
        for n in range(pnode.get("nout", 0)):
            self.add_port('tail', n);
        self.attrs = attrs

    @property
    def display_name(self):
        if self.name:
            return "[%s]"%self.name
        return "(unnamed)"

    def add_port(self, end, ident):
        '''
        Add a port of end "head" or "tail" and ident (number).
        '''
        self.ports[end].add(ident)

    def dot_name(self, port=None):
        return self.tn.replace(":","_")

    def dot_label_one(self, v, recur=True):
        if isinstance(v,list):
            siz = len(v)
            psize = min(siz, 3)
            last = ""
            if siz > psize:
                last = "..."
            if recur:
                vstr = ",".join([self.dot_label_one(vv, False) for vv in v[:psize]])
            if not recur or psize < siz:
                vstr = ",..."
            v = "list(%d):[%s]"%(siz, vstr)
            return v
        if isinstance(v,dict):
            v = "dict(%d):[%s]"%(len(v), self.dot_label_one(list(v.keys()), False))
            return v
        return str(v)


    def dot_label(self):
        ret = list()
        if "head" in self.ports:
            head = "{%s}" % ("|".join(["<in%d>%d"%(num,num) for num in sorted(self.ports["head"])]),)
            ret.append(head)

        body = [self.type, self.display_name]
        if self._params:
            for k,v in sorted(self.attrs.items()):
                v = self.dot_label_one(v)
                one = "%s = %s" % (k,v)
                body.append(one)
        body = r"\n".join(body)
        body = r"{%s}" % body
        ret.append(body)

        if "tail" in self.ports:
            tail = "{%s}" % ("|".join(["<out%d>%d"%(num,num) for num in sorted(self.ports["tail"])]),)
            ret.append(tail)

        return "{%s}" % ("|".join(ret),)


def is_string(x):
    return type(x) in [type(u""), type("")]
def is_list(x):
    return type(x) in [list]
def is_list_of_string(x):
    if not is_list(x): return False
    return all(map(is_string, x))

def dotify(edge_dat, attrs, params=True, services=True, graph_options=dict(rankdir="LR")):
    '''
    Return GraphViz text.

    If attrs is a dictionary, append to the node a list of its items.

    If params is True, show the attributes.

    If services is True, include non DFP node components.

    '''


    nodes = dict()
    
    # If node data has special _pnode item, premake its node
    for tn, nattrs in attrs.items():
        if '_pnode' in nattrs:
            nodes[tn] = Node(tn, params, **nattrs)
            

    def get(edge, end):
        try:
            tn = edge[end]["node"]
        except KeyError:
            print(f'{end=}')
            print(json.dumps(edge[end], indent=4))
            raise
        try:
            n = nodes[tn]
        except KeyError:
            n = Node(tn, params, **attrs.get(tn, {}))
            nodes[tn] = n
        p = edge[end].get("port",0)
        n.add_port(end, p)
        return n,p
    
    rankdir = graph_options.get("rankdir", "LR")
    if rankdir == "TB":
        tc = ":s"
        hc = ":n"
    else:
        tc = ":e"
        hc = ":w"
        


    edges = list()
    for edge in edge_dat:
        t, tp = get(edge, "tail")
        h, hp = get(edge, "head")
        e = '"%s":out%d%s -> "%s":in%d%s' % (t.dot_name(), tp, tc, h.dot_name(), hp, hc)
        edges.append(e);

    # Try to find non DFP node components referenced.
    if services:
        for tn,n in list(nodes.items()):
            for k,v in n.attrs.items():
                tocheck = None
                if is_string(v):
                    tocheck = [v]
                if is_list_of_string(v):
                    tocheck = v
                if not tocheck:
                    continue
                for maybe in tocheck:
                    if maybe not in attrs:
                        continue

                    cn = nodes.get(maybe,None);
                    if cn is None:
                        cn = Node(maybe, params, **attrs.get(maybe, {}))
                        nodes[maybe] = cn

                    e = '"%s" -> "%s"[style=dashed,color=gray]' % (n.dot_name(), cn.dot_name())
                    edges.append(e)

    ret = ["digraph pgraph {"]
    ret += [f'{key}={val};' for key,val in graph_options.items()]
    ret += ["\tnode[shape=record];"]
    for nn,node in sorted(nodes.items()):
        nodestr = '\t"%s"[label="%s"];' % (node.dot_name(), node.dot_label())
        ret.append(nodestr)
    for e in edges:
        ret.append("\t%s;" % e)
    ret.append("}")
    return '\n'.join(ret);


# def jsonnet_try_path(path, rel):
#     if not rel:
#         raise RuntimeError('Got invalid filename (empty string).')
#     if rel[0] == '/':
#         full_path = rel
#     else:
#         full_path = os.path.join(path, rel)
#     if full_path[-1] == '/':
#         raise RuntimeError('Attempted to import a directory')

#     if not os.path.isfile(full_path):
#         return full_path, None
#     with open(full_path) as f:
#         return full_path, f.read()


# def jsonnet_import_callback(path, rel):
#     paths = [path] + os.environ.get("WIRECELL_PATH","").split(":")
#     for maybe in paths:
#         try:
#             full_path, content = jsonnet_try_path(maybe, rel)
#         except RuntimeError:
#             continue
#         if content:
#             return full_path, content
#     raise RuntimeError('File not found')



def resolve_path(obj, dpath):
    '''
    Select out a part of obj based on a "."-separated path.  Any
    element of the path that looks like an integer will be cast to
    one assuming it indexes an array.
    '''
    if not dpath:
        return obj
    if dpath == '.':
        return obj

    dpath = dpath.split('.')
    for one in dpath:
        if not one:
            break
        try:
            one = int(one)
        except ValueError:
            pass
        obj = obj[one]

    return obj

def uses_to_params(uses):
    '''
    Given a list of nodes, return a dictionary of their "data" entries
    keyed by 'type' or 'type:name'
    '''
    ret = dict()
    for one in uses:
        if type(one) != dict:
            log.debug (f'{type(one)}, {one}')
        tn = one[u"type"]
        if "name" in one and one['name']:
            tn += ":" + str(one["name"])
        data = one.get("data", {})
        if "_pnode" in one:
            data["_pnode"] = one["_pnode"]
        ret[tn] = data
    return ret

@cli.command("dotify")
@click.option("-P","--wpath", default="", type=str,
              help="A :-separated path to add to WIRECELL_PATH")
@click.option("--dpath", default=None, type=str,
              help="A dot-delimited path into the data structure to locate a graph-like object")
@click.option("--npath", default=None, type=str,
              help="A dot-delimited path into the data structure to locate a nodes array")
@click.option("--epath", default=None, type=str,
              help="A dot-delimited path into the data structure to locate a edges array")
@click.option("--params/--no-params", default=True,
              help="Enable/disable the inclusion of contents of configuration parameters") 
@click.option("--services/--no-services", default=True,
              help="Enable/disable the inclusion 'service' (non-node) type components") 
@click.option("--graph-options", multiple=True,
              help="Graph options as key=value") 
@jsonnet_loader("in-file")
@click.argument("out-file")
@click.pass_context
def cmd_dotify(ctx, wpath, dpath, npath, epath, params, services, graph_options, in_file, out_file):
    '''
    Convert a WCT cfg to a GraphViz dot or rendered file.

    The config file may be JSON or Jsonnet and must provide an array
    of graph "nodes" and an array of graph "edges".

    A JSON pointer data path to a graph data structure embedded in a
    larger structure may be specified with --dpath DPATH.

    By default, a wire-cell job configuration object is assumed to
    hold the graph with a list of nodes in an array at DPATH and
    with the final node in the array providing a list of edges at
    DPATH.-1.data.edges.

    An arbitrary node array may be specified at --npath NPATH.

    An arbitrary edge array may be specified at --epath EPATH.

    Example bash command assuming WIRECELL_PATH properly set

      $ wirecell-pgraph dotify mycfg.jsonnet mycfg.pdf

    Or piecewise

      $ wcsonnet mycfg.jsonnet > mycfg.json

      $ wirecell-pgraph dotify mycfg.json mycfg.dot

      $ dot -Tpdf -o mycfg.pdf mycfg.dot

    The arguments -A/--tla, -J/--jpath are only valid for an input
    file in Jsonnet format.

    Note, nodes can not currentlybe drawn to reflect configured number of ports
    but only numbered by existing edges.  This can hide mistakes due missing
    edges.  The required information is lost as part of the pgraph.main() call.
    '''
    wirecell_path = os.environ.get("WIRECELL_PATH","")
    if wirecell_path:
        wpath = wirecell_path + ":" + wpath
    os.environ["WIRECELL_PATH"] = wpath

    try: 
        dat = resolve_path(in_file, dpath)
    except Exception:
        click.echo('failed to resolve path "%s" in object:\n' % (dpath))
        sys.exit(1)

    if any ((npath, epath)):
        uses = resolve_path(dat, npath)
        edges = resolve_path(dat, epath)
    else:                       # wct cfg
        uses = dat
        edges = dat[-1]["data"]["edges"]

    gopts = dict(rankdir="LR")
    if graph_options:
        gopts = dict()
        for go in graph_options:
            k,v = go.split("=",1)
            gopts[k]=v

    attrs = uses_to_params(uses)
    dtext = dotify(edges, attrs, params, services, gopts)
    ext = os.path.splitext(out_file)[1][1:]
    dot = "dot -T %s -o %s" % (ext, out_file)
    proc = subprocess.Popen(dot, shell=True, stdin = subprocess.PIPE)
    proc.communicate(input=dtext.encode("utf-8"))
    return

# --- importify: graph the Jsonnet "import" DAG -------------------------------
#
# Jsonnet import/importstr/importbin take STRING LITERALS only (computed import
# paths are forbidden by the grammar), so a static ("lexical") scan yields the
# complete import graph.  The "eval" mode instead hooks the Python jsonnet
# import callback so only imports that are actually forced (jsonnet is lazy) are
# recorded.  Both resolve paths the way jsonnet does: relative to the importing
# file's directory first, then the search paths (WIRECELL_PATH / -J / -P).

# import / importstr / importbin followed by a single- or double-quoted string.
_IMPORT_RE = re.compile(r'\b(import|importstr|importbin)\s+("([^"]*)"|\'([^\']*)\')')
_LINE_COMMENT_RE = re.compile(r'(//|#).*$', re.MULTILINE)
_BLOCK_COMMENT_RE = re.compile(r'/\*.*?\*/', re.DOTALL)


def scan_imports(path):
    '''Return list of (kind, import-string) literals appearing in file at path.'''
    with open(path, encoding="utf-8") as fp:
        text = fp.read()
    text = _BLOCK_COMMENT_RE.sub('', text)
    text = _LINE_COMMENT_RE.sub('', text)
    return [(m.group(1), m.group(3) if m.group(3) is not None else m.group(4))
            for m in _IMPORT_RE.finditer(text)]


def resolve_import(importer, rel, paths):
    '''Resolve import string `rel` seen in file `importer` against jsonnet rules.

    Tries the importing file's directory first then each search path, exactly as
    jsio.ImportCallback does.  Returns an absolute path string or None.
    '''
    for base in [os.path.dirname(importer)] + [str(p) for p in paths]:
        try:
            full_path, content = jsio.try_path(base, rel)
        except RuntimeError:
            continue
        if content:
            return str(full_path.absolute())
    return None


def lexical_graph(root, paths):
    '''Static import graph rooted at `root`.

    Returns (edges, unresolved) where edges is a set of (src, dst, kind) with
    absolute path endpoints and unresolved is a set of (src, rel, kind).
    '''
    edges = set()
    unresolved = set()
    seen = set()
    stack = [os.path.abspath(root)]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for kind, rel in scan_imports(cur):
            dst = resolve_import(cur, rel, paths)
            if dst is None:
                unresolved.add((cur, rel, kind))
                continue
            edges.add((cur, dst, kind))
            if dst not in seen:
                stack.append(dst)
    return edges, unresolved


def eval_graph(root, paths, jkwds):
    '''Import graph of only the imports jsonnet actually evaluates (lazily).

    Evaluates `root` with a recording import callback to collect the set of
    forced imports, then draws the lexical edges restricted to that set.  A
    lexical edge is kept only when its target was actually evaluated, so
    imported-but-never-forced branches (and their subtrees) are excluded.
    Returns (edges, evaluated-node-set).
    '''
    root = os.path.abspath(root)
    ic = jsio.ImportCallback(paths)
    jsmod = jsio.jsonnet_module()
    with open(root, encoding="utf-8") as fp:
        text = fp.read()
    jsmod.evaluate_snippet(root, text, import_callback=ic, **jkwds)

    nodes = {os.path.abspath(p) for p in ic.found} | {root}
    edges = set()
    for src in nodes:
        for kind, rel in scan_imports(src):
            dst = resolve_import(src, rel, paths)
            if dst is not None and os.path.abspath(dst) in nodes:
                edges.add((src, os.path.abspath(dst), kind))
    return edges, nodes


def importify_dot(edges, roots, extra_nodes=(), unresolved=()):
    '''Render (edges, roots) as graphviz dot text.

    edges: iterable of (src, dst, kind) absolute paths.
    roots: iterable of absolute root paths (highlighted).
    extra_nodes: nodes to include even if they have no edges.
    unresolved: iterable of (src, rel, kind) drawn as dashed red targets.
    '''
    roots = {os.path.abspath(r) for r in roots}
    nodes = set(roots) | set(os.path.abspath(n) for n in extra_nodes)
    for src, dst, _ in edges:
        nodes.add(src)
        nodes.add(dst)
    common = os.path.commonpath(list(nodes)) if nodes else os.getcwd()

    def lab(p):
        return os.path.relpath(p, common)

    lines = ["digraph jsonnet_imports {",
             '  rankdir=LR; node [shape=box, fontname="monospace", fontsize=10];']
    for n in sorted(nodes):
        attrs = ' style=filled, fillcolor="#cde7ff"' if n in roots else ''
        lines.append('  "%s" [%s];' % (lab(n), attrs.strip()))
    for src, rel, _kind in sorted(set(unresolved)):
        tgt = "%s (unresolved)" % rel
        lines.append('  "%s" [style=filled, fillcolor="#ffd0d0"];' % tgt)
        lines.append('  "%s" -> "%s" [style=dashed, color=red];' % (lab(src), tgt))
    for src, dst, kind in sorted(edges):
        style = "solid" if kind == "import" else "dashed"
        lines.append('  "%s" -> "%s" [style=%s];' % (lab(src), lab(dst), style))
    lines.append("}")
    return "\n".join(lines) + "\n"


@cli.command("importify")
@click.option("-P", "--wpath", default="", type=str,
              help="A :-separated path to add to WIRECELL_PATH")
@click.option("-J", "--jpath", multiple=True, envvar="WIRECELL_PATH",
              help="A file system path to locate Jsonnet files (repeatable)")
@click.option("-A", "--tla", multiple=True,
              help="Set a top-level argument as key=val, key=code or key=filename (eval mode)")
@click.option("-V", "--ext", multiple=True,
              help="Set an external var as key=val (eval mode)")
@click.option("-m", "--mode", type=click.Choice(["lexical", "eval"]), default="lexical",
              help="'lexical': static scan of all import literals (complete). "
                   "'eval': only imports jsonnet actually forces (lazy).")
@click.argument("in-file")
@click.argument("out-file")
@click.pass_context
def cmd_importify(ctx, wpath, jpath, tla, ext, mode, in_file, out_file):
    '''
    Graph the Jsonnet "import" DAG of a config file as GraphViz dot.

    This is the graph of files formed by import/importstr/importbin directives,
    NOT the wire-cell node graph (see "dotify" for that).

    Two modes:

      - lexical (default): a static scan of every import string literal.  This
        is the complete import graph and needs no evaluation, so it works even
        when the config would fail to evaluate.  Imports that jsonnet would
        never force (it is lazy) are still shown.  Unresolvable imports are
        drawn as dashed red edges and also reported on stderr.

      - eval: evaluate the config through the Python jsonnet import system and
        record only the imports actually forced.  Imported-but-never-used
        branches are excluded.  Requires any top-level arguments the config
        needs (pass with -A), same as dotify.

    Search paths honor WIRECELL_PATH plus any -J/--jpath and -P/--wpath.  The
    output format follows the OUT-FILE extension (.dot writes dot text, anything
    else is rendered with graphviz, e.g. .pdf/.png).  Use - for dot on stdout.

    Examples

      $ wcpy pgraph importify mycfg.jsonnet imports.pdf

      $ wcpy pgraph importify --mode eval -A input=x.npz mycfg.jsonnet imports.dot
    '''
    # Assemble search paths: -J entries (default WIRECELL_PATH), then -P/--wpath.
    raw = list(jpath)
    if wpath:
        raw.append(wpath)
    paths = jsio.wash_path(raw)

    root = str(jsio.resolve(in_file, paths))

    unresolved = ()
    extra_nodes = ()
    if mode == "lexical":
        edges, unresolved = lexical_graph(root, paths)
        for src, rel, kind in sorted(unresolved):
            log.warning('unresolved import: %s -> %s (%s)'
                        % (os.path.relpath(src), rel, kind))
    else:
        jkwds = jsio.tla_pack(tla, paths)
        jkwds.update(jsio.tla_pack(ext, paths, 'ext_'))
        try:
            edges, nodes = eval_graph(root, paths, jkwds)
        except RuntimeError as err:
            raise click.ClickException(
                "eval mode failed to evaluate %s (missing -A top-level args?):\n%s"
                % (root, err))
        extra_nodes = nodes

    dtext = importify_dot(edges, [root], extra_nodes=extra_nodes, unresolved=unresolved)

    if out_file == "-":
        click.echo(dtext, nl=False)
        return
    ext_out = os.path.splitext(out_file)[1][1:]
    if ext_out == "dot":
        with open(out_file, "w", encoding="utf-8") as fp:
            fp.write(dtext)
        return
    dot = "dot -T %s -o %s" % (ext_out, out_file)
    proc = subprocess.Popen(dot, shell=True, stdin=subprocess.PIPE)
    proc.communicate(input=dtext.encode("utf-8"))
    return


def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
