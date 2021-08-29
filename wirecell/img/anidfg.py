#!/usr/bin/env python3
'''
animage data flow graph info
'''
import os
from math import floor
from collections import namedtuple
from gvanim import Animation, render, gif

def parse_ts(ts):
    '''
    Parse "[HH:MM:SS.ms]" into a floating point number
    '''
    hh,mm,ssms = ts[1:-1].split(":")
    return float(hh)*3600 + float(mm)*60 + float(ssms)


Connect = namedtuple("Connect", "time head tail hport tport")
State = namedtuple("State", "time node state")

def parse_log(fp, group="dfg"):
    '''
    Parse a file object giving "dfg" log lines.

    Given default group and tn, lines are expected to look like:

    [HH:MM:SS.ms] D [  dfg   ] <TbbFlow:> ....

    That is, as produced by TbbFlow logging.

    This yields per line info
    '''

    for line in fp.readlines():
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) < 7:
            continue
        ts,lvl,_,grp,_ = parts[:5]
        if grp != group:
            continue

        ts = parse_ts(ts)

        if "Connect:" in parts:
            head = parts[-1]
            hport = parts[-3]
            tport = parts[-5]
            tail = parts[-7]
            yield Connect(ts, head, tail, hport, tport)
    
        if parts[5].startswith("node="):
            _, node = parts[5].split("=")
            _, state = parts[6].split("=")
            yield State(ts, node, state)

            
def generate_graph(lines, tick=0.5):
    '''
    Given lines from parse_log, return a gvanim.Animate object.

    The tick determines the frame rate and is in units of the .time
    value of lines.
    '''

    ga = Animation()

    last_time = 0

    for one in lines:
            
        if isinstance(one, Connect):
            ga.add_edge(one.tail, one.head)
            last_time = one.time

        if isinstance(one, State):
            if one.state == "enter":
                ga.highlight_node(one.node)
            if one.state == "exit":
                ga.add_node(one.node)
            if one.time - last_time > tick:
                ga.next_step()
                last_time = one.time

    ga.next_step()
    return ga


                

def render_graph(ga, outfile, size=1024, tempfiletype='png'):
    '''
    Render the gv animation
    '''
    base = os.path.splitext(outfile)[0]

    graphs = ga.graphs()
    files = render(graphs, base, tempfiletype, size=size)
    gif(files, base, 500, size=size)
