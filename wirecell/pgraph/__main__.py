#!/usr/bin/python3
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

def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
