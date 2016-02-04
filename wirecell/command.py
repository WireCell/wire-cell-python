#!/usr/bin/env python
'''Python interface to the wire-cell command.

Configuration:

* WC file is a wire-cell configuration file in JSON format.

* WC JSON is the contents of a WC file.

* A configlist is the Python data structure representing a WC JSON
  string.  It is a list of dictionaries with keys "type", "name" and
  "data" which holds a type-specific data structure.

* A configdict is a dictionary keyed by tuple(type,name) with a value
  that of the corresponding "data" value.

The C++ Wire Cell components which are configurable provide a default,
hard-coded configuration "data" structure.

'''
from collections import defaultdict
import json
import subprocess

class Config:

    formats = 'wcjson wcfile configlist configdict'.split()

    def __init__(self, **kwds):
        self._cfg = defaultdict(dict)
        self.load(**kwds)

    def load(self, **kwds):
        for form in self.formats:
            obj = kwds.get(form,None)
            if not obj: continue
            meth = getattr(self, 'load_'+form)
            meth(obj)

    def load_wcjson(self, string):
        cl = json.loads(string)
        self.load_configlist(cl)

    def load_wcfile(self, filename):
        self.load_wcjson(open(filename).read())

    def load_configlist(self, cfglist):
        for d in cfglist:
            self.load_one(d['type'], d.get('name',''), d['data'])

    def load_configdict(self, cfgdict):
        for (t,n),d in self._cfg.items():
            self.load_one(t,n,d)

    def get(self, type, name=''):
        return self._cfg[(type,name)]

    def load_one(self, type, name, data):
        val = self.get(type,name)
        val.update(data)
        self._cfg[(type,name)] = val
        
        
    def merge(self, other):
        '''
        Merge other into copy of self and return new Config object
        '''
        ret = Config()
        for (t,n),d in self._cfg.items():
            ret.load_one(t,n,d)
        if other:
            for (t,n),d in other._cfg.items():
                ret.load_one(t,n,d)
        return ret

    def wcjson(self):
        '''
        Return wire-cell compatible JSON representation.
        '''
        dat = list()
        for (t,n),d in self._cfg.items():
            dat.append(dict(type=t, name=n, data=d))
        return json.dumps(dat)

    def __str__(self):
        return self.wcjson()

class WireCell:
    '''Python interface to wire-cell executable.

    Beware, this trusts its input!  In particular do not let
    <executable> be set by untrusted sources.
    '''

    default_plugins = ['WireCellAlg', 'WireCellGen', 'WireCellApps', 'WireCellTbb']

    def __init__(self, executable='wire-cell', **kwds):
        self.prog = executable
        self.plugins = kwds.pop('plugins', self.default_plugins)
        self.config = Config(**kwds)
        self.app = kwds.get('app',None)

    def cmdline(self, app=None):
        '''
        Return command line string
        '''
        parts = ['%s -c /dev/stdin' % self.prog]
        parts += ['-p %s'%p for p in self.plugins]
        app = app or self.app
        if app:
            parts.append('-a %s' % app)
        return ' '.join(parts)

    def __call__(self, app=None, config=None):
        '''Run wire-cell.

        If app is not given then fall back to one given to constructor.

        If any config is given it is merged with any given to constructor.

        '''
        cmd = self.cmdline(app)
        #print 'CMD:\n%s'%cmd
        cfg = self.config.merge(config).wcjson()
        #print 'CFG:\n%s'%cfg        
        proc = subprocess.Popen(cmd, shell=True, 
                                stdin = subprocess.PIPE,
                                stdout = subprocess.PIPE,
                                stderr = subprocess.PIPE)
        out,err= proc.communicate(cfg)
        #print 'OUT:\n%s'%out
        #print 'ERR:\n%s'%err
        return out,err
        
    def component_configlist(self, *components):
        '''
        Return a config list giving configuration for components
        '''
        config = Config()
        config.load_one("ConfigDumper","", data=dict(components = components))
        o,e = self("ConfigDumper", config=config)
        return json.loads(o)
        
