from common import bounding_box

def wire_summary(all_wires):
    for iplane in range(3):
        wires = [w for w in all_wires if w.plane == iplane]
        chans = [w.ch for w in wires]
        wids = [w.wip for w in wires]
        bb = bounding_box(wires)
        minpt, maxpt = bb.tail, bb.head
        
        fmt = "plane %s: %d wires [%d - %d], %d chans [%04d - %04d], minpt=(%+.1f, %+.1f, %+.4f), maxpt=(%+.1f, %+.1f, %+.1f)"
        dat = ("UVW"[iplane], len(wires), min(wids), max(wids), len(chans), min(chans), max(chans),
                   minpt[0],minpt[1],minpt[2],maxpt[0],maxpt[1],maxpt[2])
        print fmt%dat

