#!/usr/bin/env python
'''
Tests for wirecell.util.logtimes.
'''
from wirecell.util import logtimes

SAMPLE = '''\
[17:28:47.191] I [ timer  ] Timer: 2.266 wall-sec, 2.477 core-sec:  (WireCell::Sio::FrameFileSink) "roll.npz"
[17:28:47.191] I [ timer  ] Timer: 0.705 wall-sec, 1.977 core-sec:  (WireCell::SPNG::KernelConvolve) "tpc0_group_v0c2f1f0"
[17:28:47.192] I [ timer  ] Timer: 0.000 wall-sec, 0.001 core-sec:  (WireCell::SPNG::FanoutTensorSets) "tpc0_bypass"
'''


def test_parse_line():
    line = ('[17:28:47.191] I [ timer  ] Timer: 2.266 wall-sec, '
            '2.477 core-sec:  (WireCell::Sio::FrameFileSink) "roll.npz"')
    one = logtimes.parse_line(line)
    assert one == {
        "wall-sec": 2.266,
        "core-sec": 2.477,
        "class": "WireCell::Sio::FrameFileSink",
        "instance": "roll.npz",
    }


def test_parse_line_nonmatch():
    assert logtimes.parse_line("some unrelated log line") is None
    assert logtimes.parse_line("") is None


def test_parse_lines():
    got = logtimes.parse_lines(SAMPLE.splitlines())
    assert len(got) == 3
    assert got[0]["class"] == "WireCell::Sio::FrameFileSink"
    assert got[1]["wall-sec"] == 0.705
    assert got[1]["core-sec"] == 1.977
    assert got[2]["instance"] == "tpc0_bypass"
    assert got[2]["wall-sec"] == 0.0


def test_parse_lines_skips_noise():
    lines = SAMPLE.splitlines() + ["not a timer line", ""]
    got = logtimes.parse_lines(lines)
    assert len(got) == 3


def test_parse_file(tmp_path):
    p = tmp_path / "wire-cell.log"
    p.write_text(SAMPLE)
    got = logtimes.parse_file(str(p))
    assert len(got) == 3
    assert all(set(d) == {"wall-sec", "core-sec", "class", "instance"}
               for d in got)
