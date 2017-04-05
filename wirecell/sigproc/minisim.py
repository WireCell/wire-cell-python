#!/usr/bin/env python
'''
A simple drift simulation using response functions.   No diffusion.
'''

from .. import units
import numpy
import collections
from math import radians, sin, cos

# ie, Geant4 hits.
Hit = collections.namedtuple("Hit", "x y z t q")

def dir_yz(deg):
    return (sin(radians(deg)), cos(radians(deg)))

class Minisim(object):
    '''
    Simulate the response to a distribution of charge.

    todo:

        - for each hit, find all rf's for the hits impact, use each
          region number as offset to hit's wire number, fill frame.
    '''

    defaults = dict(
        nticks=9600,
        nwires=(2400,2400,3456),
        pitch=3.0*units.mm,
        impact=0.3*units.mm,
        velocity=1.6*units.mm/units.us,
        tick=0.5*units.us,      # digitization tick
        )
    def __init__(self, pib, **kwds):
        '''
        Create a mini simulation.

        - pib :: a response.PlaneImpactBlocks object

        '''
        self.pib = pib
        self._cfg = dict(self.defaults, **kwds)

        self.wire_yz_dir = numpy.asarray((dir_yz(30), dir_yz(150), dir_yz(90)))
        self.wire_yz_pit = numpy.asarray((dir_yz(-60), dir_yz(60), dir_yz(0)))
        self.wire_yz_off = numpy.asarray(((0.0, 0.0),
                                          (0.0, 0.0),
                                          (0.0, 0.5*self.pitch)))

        self.frame = None       # becomes 3-tuple
        self.clear_frame()      # prime

    def __getattr__(self, key):
        return self._cfg[key]

    def wire_space(self, hits):
        '''
        Return the location of a hit in terms of (wire number, impact).
        '''
        ret = list()

        hit_yz = hits[:,1:3]
        for iplane in range(3):
            yz = hit_yz - self.wire_yz_off[iplane]
            pit = numpy.dot(yz, self.wire_yz_pit[iplane]) # distance along pitch from center
            pwidth = self.pitch * self.nwires[iplane]/2
            pit += pwidth       # distance along pitch from wire 0
            wiredist = pit/self.pitch
            wire = numpy.asarray(numpy.round(wiredist), dtype=int)
            impact = numpy.round((wiredist-wire)/self.impact) * self.impact
            ret.append(numpy.vstack((wire, impact)).T)
        return ret
        

    def apply_block(self, block, iplane, ich, time):
        '''
        Apply block array to plane array with ich and itick offsets
        '''
        dch = block.shape[0]//2
        chm = int(max(ich-dch, 0))
        chp = int(min(ich+dch, self.nwires[iplane]-1))
        #print 'ch:',dch, chm, chp

        tfinebin_min = int(time/self.pib.tbin)
        tfinebin_jmp = int(self.tick/self.pib.tbin)
        #print 'time:',time,tfinebin_min, tfinebin_jmp

        sampled = block[:,tfinebin_min::tfinebin_jmp]
        #print 'sampled:',sampled.shape

        tsampbin_num = len(sampled[0])
        tsampbin_min = int(time/self.tick)
        tsampbin_max = min(tsampbin_min+tsampbin_num, len(self.frame[iplane][0]))
        print 'samp bin:',tsampbin_num, tsampbin_min, tsampbin_max*self.tick/units.us

        self.frame[iplane][chm:chp+1, tsampbin_min:tsampbin_max] = sampled

        
    def clear_frame(self):
        '''
        Initialize a new frame.
        '''
        frame = list()
        for nwires in self.nwires:
            frame.append(numpy.zeros((nwires, self.nticks)))
        self.frame = tuple(frame)
        return

    def add_hits(self, hits):
        '''
        Add response to hits to current frame.
        '''
        nearest_chimps = self.wire_space(hits)
        for iplane, chimps in enumerate(nearest_chimps):
            for ihit, (ch,imp) in enumerate(chimps):
                x,y,z,q,t = hit = hits[ihit]
                block = self.pib.region_block("uvw"[iplane], imp)
                self.apply_block(block, iplane, ch, (t-self.pib.tmin) + (x-self.pib.xstart)/self.velocity)
            
            
            
