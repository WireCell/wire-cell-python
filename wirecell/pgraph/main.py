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
    def __init__(self, tn):
        self.tn = tn
        tn = tn.split(":")
        self.type = tn[0]
        try:
            self.name = tn[1]
        except IndexError:
            self.name = ""
        self.ports = defaultdict(set);

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

        body = r"{%s\n%s}" % (self.type, self.name)
        ret.append(body)

        if self.ports.has_key("tail"):
            tail = "{%s}" % ("|".join(["<out%d>%d"%(num,num) for num in sorted(self.ports["tail"])]),)
            ret.append(tail)

        return "{%s}" % ("|".join(ret),)


def dotify(dat):
    '''
    Return GraphViz text 
    '''
    nodes = dict()
    def get(edge, end):
        tn = edge[end]["node"]
        try:
            n = nodes[tn]
        except KeyError:
            n = Node(tn)
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



@cli.command("dotify")
@click.argument("json-file")
@click.argument("out-file")
@click.pass_context
def cmd_dotify(ctx, json_file, out_file):
    '''
    Convert a JSON file for a WCT job configuration based on the Pgraph app into a dot file.
    '''
    if json_file.endswith(".jsonnet"):
        import _jsonnet
        jtext = _jsonnet.evaluate_file(json_file, import_callback=jsonnet_import_callback)
    else:
        jtext = open(json_file).read()

    for cfg in json.loads(jtext):
        if cfg["type"] != "Pgrapher":
            continue;

        edges = cfg["data"]["edges"]
        dtext = dotify(edges)
        ext = os.path.splitext(out_file)[1][1:]
        dot = "dot -T %s -o %s" % (ext, out_file)
        proc = subprocess.Popen(dot, shell=True, stdin = subprocess.PIPE)
        proc.communicate(input=dtext)
        return

def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
