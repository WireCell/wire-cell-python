#!/usr/bin/env python3
'''
Parse Wire-Cell "Timer" log lines.

Wire-Cell components emit timing summaries via the "timer" logger.  A typical
line looks like:

    [17:28:47.191] I [ timer  ] Timer: 2.266 wall-sec, 2.477 core-sec:  (WireCell::Sio::FrameFileSink) "roll.npz"

This module extracts the wall-sec, core-sec, class name and instance name from
each such line.
'''

import re

# Matches the payload of a "Timer:" log line.  Leading log decoration (time
# stamp, level, logger name) is ignored so lines may be matched regardless of
# the log formatting in use.
_timer_re = re.compile(
    r'Timer:\s*'
    r'(?P<wall>[-+0-9.eE]+)\s*wall-sec,\s*'
    r'(?P<core>[-+0-9.eE]+)\s*core-sec:\s*'
    r'\((?P<cls>[^)]*)\)\s*'
    r'"(?P<inst>[^"]*)"'
)


def parse_line(line):
    '''
    Parse a single log line.

    Return a dict with keys "wall-sec", "core-sec", "class" and "instance" or
    None if the line is not a Timer line.
    '''
    m = _timer_re.search(line)
    if not m:
        return None
    return {
        "wall-sec": float(m.group("wall")),
        "core-sec": float(m.group("core")),
        "class": m.group("cls"),
        "instance": m.group("inst"),
    }


def parse_lines(lines):
    '''
    Parse an iterable of log lines, returning a list of dicts for each Timer
    line found.  Non-Timer lines are silently skipped.
    '''
    out = []
    for line in lines:
        one = parse_line(line)
        if one is not None:
            out.append(one)
    return out


def parse_file(fname):
    '''
    Parse a log file given by name, returning a list of Timer dicts.
    '''
    with open(fname) as fp:
        return parse_lines(fp)
