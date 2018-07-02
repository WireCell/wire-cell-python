#!/usr/bin/python
'''
Fixme: make this into a proper click main
'''
import os
import sys
import json
from collections import defaultdict
import subprocess

import click

from wirecell import units

@click.group("pgraph")
@click.pass_context
def cli(ctx):
    '''
    Wire Cell Signal Processing Features
    '''

class Node (object):
    def __init__(self, tn, **attrs):
        self.tn = tn
        tn = tn.split(":")
        self.type = tn[0]
        try:
            self.name = tn[1]
        except IndexError:
            self.name = ""
        self.ports = defaultdict(set);
        self.attrs = attrs

    def add_port(self, end, ident):
        '''
        Add a port of end "head" or "tail" and ident (number).
        '''
        self.ports[end].add(ident)

    def dot_name(self, port=None):
        return self.tn.replace(":","_")

    def dot_label(self):
        ret = list()
        if self.ports.has_key("head"):
            head = "{%s}" % ("|".join(["<in%d>%d"%(num,num) for num in sorted(self.ports["head"])]),)
            ret.append(head)

        body = [self.type, self.name]
        for k,v in sorted(self.attrs.items()):
            if isinstance(v,list):
                v = "list(%d)"%len(v)
            if isinstance(v,dict):
                v = "dict(%d)"%len(v)
            one = "%s = %s" % (k,v)
            body.append(one)
        body = r"\n".join(body)
        body = r"{%s}" % body
        ret.append(body)

        if self.ports.has_key("tail"):
            tail = "{%s}" % ("|".join(["<out%d>%d"%(num,num) for num in sorted(self.ports["tail"])]),)
            ret.append(tail)

        return "{%s}" % ("|".join(ret),)


def dotify(dat, attrs):
    '''
    Return GraphViz text.  If attrs is a dictionary, append to the node a list of its items.
    '''
    nodes = dict()
    def get(edge, end):
        tn = edge[end]["node"]
        try:
            n = nodes[tn]
        except KeyError:
            n = Node(tn, **attrs.get(tn, {}))
            nodes[tn] = n
        p = edge[end].get("port",0)
        n.add_port(end, p)
        return n,p
    
    edges = list()
    for edge in dat:
        t,tp = get(edge, "tail")
        h,hp = get(edge, "head")
        e = '%s:out%d -> %s:in%d' % (t.dot_name(),tp, h.dot_name(),hp)
        edges.append(e);

    ret = ["digraph pgraph {",
           "rankdir=LR;",
           "\tnode[shape=record];"]
    for nn,node in sorted(nodes.items()):
        ret.append('\t%s[label="%s"];' % (node.dot_name(), node.dot_label()))
    for e in edges:
        ret.append("\t%s;" % e)
    ret.append("}")
    return '\n'.join(ret);

def jsonnet_try_path(path, rel):
    if not rel:
        raise RuntimeError('Got invalid filename (empty string).')
    if rel[0] == '/':
        full_path = rel
    else:
        full_path = os.path.join(path, rel)
    if full_path[-1] == '/':
        raise RuntimeError('Attempted to import a directory')

    if not os.path.isfile(full_path):
        return full_path, None
    with open(full_path) as f:
        return full_path, f.read()


def jsonnet_import_callback(path, rel):
    paths = [path] + os.environ.get("WIRECELL_PATH","").split(":")
    for maybe in paths:
        try:
            full_path, content = jsonnet_try_path(maybe, rel)
        except RuntimeError:
            continue
        if content:
            return full_path, content
    raise RuntimeError('File not found')



def resolve_path(obj, jpath):
    '''
    Select out a part of obj based on a "."-separated path.  Any
    element of the path that looks like an integer will be cast to
    one assuming it indexes an array.
    '''
    jpath = jpath.split('.')
    for one in jpath:
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
    Given a list of "uses", return a dictionary of their "data" entries keyed by type:name
    '''
    ret = dict()
    for one in uses:
        tn = one["type"]
        if one.has_key("name"):
            tn += ":" + one["name"]
        ret[tn] = one.get("data", {})
    return ret

@cli.command("dotify")
@click.option("--jpath", default="-1",
              help="A dot-delimited path into the JSON to locate a graph-like object")
@click.option("--params/--no-params", default=True,
              help="Enable/disable the inclusion of contents of configuration parameters") 
@click.argument("json-file")
@click.argument("out-file")
@click.pass_context
def cmd_dotify(ctx, jpath, params, json_file, out_file):
    '''
    Convert a JSON file for a WCT job configuration based on the
    Pgraph app into a dot file.

    Use jpath to apply this function to a subset of what the input
    file compiles to.  Default is "-1" which usually works well for a
    full configuration sequence in which the last element is the
    config for a Pgrapher.
    '''
    if json_file.endswith(".jsonnet"):
        import _jsonnet
        jtext = _jsonnet.evaluate_file(json_file, import_callback=jsonnet_import_callback)
    else:
        jtext = open(json_file).read()

    dat = json.loads(jtext)
    try: 
        cfg = resolve_path(dat, jpath)
    except Exception:
        click.echo("failed to resolve path in object:\n")
        click.echo(json.dumps(cfg, indent=4))
        sys.exit(1)

    if cfg["type"] not in ["Pgrapher", "Pnode"]:
        click.echo('Object must be of "type" Pgrapher or Pnode, got "%s"' % cfg["type"])
        sys.exit(1)

    if cfg["type"] == "Pgrapher":    # the Pgrapher app holds edges in "data" attribute
        edges = cfg["data"]["edges"] 
        uses = dat                   # if Pgrapher, then original is likely the full config sequence.
    else:
        edges = cfg["edges"] # Pnodes have edges as top-level attribute
        uses = cfg["uses"]
    attrs = dict()
    if params:
        attrs = uses_to_params(uses)
    dtext = dotify(edges, attrs)
    ext = os.path.splitext(out_file)[1][1:]
    dot = "dot -T %s -o %s" % (ext, out_file)
    proc = subprocess.Popen(dot, shell=True, stdin = subprocess.PIPE)
    proc.communicate(input=dtext)
    return

def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
