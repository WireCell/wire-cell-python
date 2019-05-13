#!/usr/bin/env python3
'''
Handle cluster graphs.

A cluster graph is a networkx graph with nodes holding a letter "type
code" (c,w,b,s,m) and a code-dependent data structure
'''

def match_dict(have, want):
    '''
    return True if all keys of want are in have and all their values are equal.
    '''
    for w in want:
        if w not in have:
            return False
        if have[w] != want[w]:
            return False
    return True

class ClusterMap(object):
    '''
    Add some indexing and lookups to meta data on cluster graph vertices
    '''

    def __init__(self, gr):
        self.gr = gr
        self._id2ch = dict()
        self._pi2ch = dict()
        self._cs2wire = dict()
        self._wip2wire = dict()
        self._wid2wire = dict()

        for node, data in gr.nodes.data():
            if data['code'] == 'c':
                self._id2ch[data['ident']] = node
                self._pi2ch[(data['wpid'], data['index'])] = node;
                continue;
            if data['code'] == 'w':
                self._cs2wire[(data['chid'], data['seg'])] = node
                self._wip2wire[(data['wpid'], data['index'])] = node;
                self._wid2wire[(data['wpid'], data['ident'])] = node;
                continue

    def channel(self, key):
        '''
        Return a channel node by a key.  If key is scalar it is a
        channel ident number, else assumed to be a pair of (wpid,
        index).
        '''
        if type(key) == tuple:
            return self._pi2ch[key];
        return self._id2ch[key];
        
    def wire_chanseg(self, chan, seg):
        '''
        Return a wire node by its channel and segment
        '''
        return self._cs2wire[(chan,seg)]
    
    def wire_wip(self, wpid, wip):
        '''
        Return a wire node by its wire-in-plane number in the given wire-plane ID
        '''
        return self._wip2wire[(wpid, wip)];

    def wire_wid(self, wpid, wid):
        '''
        Return a wire node by its wire-ident and wire-plane ID.
        '''
        return self._wid2wire[(wpid, wip)];


    def find(self, typecode=None, **kwds):
        '''
        Return nodes with data matching kwds.  If typecode is given,
        only consider nodes of that type.
        '''
        ret = list()
        for node,data in self.gr.nodes.data():
            if typecode and self.gr.nodes[node]['code'] != typecode:
                continue;
            if not match_dict(data,kwds):
                continue
            ret.append(node)
        return ret


    def nodes_oftype(self, typecode):
        '''
        Return a list of nodes of given type code
        '''
        return [n for n,d in self.gr.nodes.data() if d['code'] == typecode]

    def neighbors_oftype(self, node, typecode):
        '''
        Return all connected nodes of the given node and given type.
        '''
        ret = list()
        for nn in self.gr[node]:
            if self.gr.nodes[nn]['code'] == typecode:
                ret.append(nn)
        return ret
