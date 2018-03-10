#!/usr/bin/python
'''
Fixme: make this into a proper click main
'''

import sys
import json
from collections import defaultdict

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

def main(filename):
    edges = json.load(open(filename))[-1]["data"]["edges"]
    print dotify(edges)

if '__main__' == __name__:
    main(sys.argv[1])

