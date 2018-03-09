#!/usr/bin/python
'''
Fixme: make this into a proper click main
'''

import sys
import json

edges = json.load(open(sys.argv[1]))[-1]["data"]["edges"]

print ("digraph pgraph {");
for edge in edges:
    t = edge["tail"]["node"].replace(":"," ")
    h = edge["head"]["node"].replace(":"," ")
    print ('\t"%s" -> "%s";' % (t,h));
print ("}");

