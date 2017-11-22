#!/usr/bin/env python
'''
This module generates a DUNE connectivity graph.
'''
from collections import namedtuple
import numpy
import networkx

# This matrix connects the:
# 
#    (chip row, channel column)
#
# in an FEMBwith the conductor expressed as a pair of values:
# 
#   (layer letter, attachement number)
#
# The layer is expressed as a letter in "uvw", the attachment number
# is a one-based count of the logical spot where the conductor
# attaches to the top of the APA looking at the face holding the
# conductor and ordered left to right.
#
chip_channel_layer_spot_matrix = numpy.array([
    [('u', 19), ('u', 17), ('u', 15), ('u', 13), ('u', 11), ('v', 19),
     ('v', 17), ('v', 15), ('v', 13), ('v', 11), ('w', 23), ('w', 21),
     ('w', 19), ('w', 17), ('w', 15), ('w', 13)],
    [('u', 9), ('u', 7), ('u', 5), ('u', 3), ('u', 1), ('v', 9),
     ('v', 7), ('v', 5), ('v', 3), ('v', 1), ('w', 11), ('w', 9),
     ('w', 7), ('w', 5), ('w', 3), ('w', 1)],
    [('w', 14), ('w', 16), ('w', 18), ('w', 20), ('w', 22), ('w', 24),
     ('v', 12), ('v', 14), ('v', 16), ('v', 18), ('v', 20), ('u', 12),
     ('u', 14), ('u', 16), ('u', 18), ('u', 20)],
    [('w', 2), ('w', 4), ('w', 6), ('w', 8), ('w', 10), ('w', 12),
     ('v', 2), ('v', 4), ('v', 6), ('v', 8), ('v', 10), ('u', 2),
     ('u', 4), ('u', 6), ('u', 8), ('u', 10)],
    [('u', 29), ('u', 27), ('u', 25), ('u', 23), ('u', 21), ('v', 29),
     ('v', 27), ('v', 25), ('v', 23), ('v', 21), ('w', 35), ('w', 33),
     ('w', 31), ('w', 29), ('w', 27), ('w', 25)],
    [('u', 39), ('u', 37), ('u', 35), ('u', 33), ('u', 31), ('v', 39),
     ('v', 37), ('v', 35), ('v', 33), ('v', 31), ('w', 47), ('w', 45),
     ('w', 43), ('w', 41), ('w', 39), ('w', 37)],
    [('w', 26), ('w', 28), ('w', 30), ('w', 32), ('w', 34), ('w', 36),
     ('v', 22), ('v', 24), ('v', 26), ('v', 28), ('v', 30), ('u', 22),
     ('u', 24), ('u', 26), ('u', 28), ('u', 30)],
    [('w', 38), ('w', 40), ('w', 42), ('w', 44), ('w', 46), ('w', 48),
     ('v', 32), ('v', 34), ('v', 36), ('v', 38), ('v', 40), ('u', 32),
     ('u', 34), ('u', 36), ('u', 38), ('u', 40)]], dtype=object)

def flatten_cclsm(mat = chip_channel_layer_spot_matrix):
    '''
    Flatten an ASIC channel X number matrix to a dictionary keyed by
    (plane letter, local wire attachment number (1-48 or 1-40).  Value
    is a tuple (ichip, ich) with ichip:{1-8} and ich:{1-16}
    '''
    ret = dict()
    for ichip, row in enumerate(mat):
        for ich, cell in enumerate(row):
            cell = tuple(cell)
            ret[cell] = (ichip+1, ich+1)
    return ret


ApaFaceParams = namedtuple("ApaFaceParams", ["nlayers", "nboards"]);
ApaBoardParams = namedtuple("ApaBoardParams", ["nchips", "nchanperchip"])
ApaDaqParams = namedtuple("ApaDaqParams", ["nwibs", "nconnperwib"])
ApaParams = namedtuple("ApaParams", ["nfaces", "face", "board", "daq"])

default_apa_params = ApaParams(
    nfaces = 2,
    face = ApaFaceParams(3, 10),
    board = ApaBoardParams(8, 16),
    daq = ApaDaqParams(5, 4)
)

ApaMakers = namedtuple("ApaMakers", ["apa", "anode", "crate", "face"...]

class ApaConnectivity(object):
    '''
    Provide methods to enumerate connectivity
    '''
    def __init__(self,  params = default_apa_params):
        self.p = params
        self.nboards = self.p.face.nboards*self.p.nfaces
        self.nchips = self.nboards * self.p.board.nchips
        self.nchans = self.nchips*self.p.board.nchanperchip

        # List of indicies to boards in two ways: [face,board] and WIB
        # [conn,slot].  A very smart layout convention adopted by the
        # engineers let us do this so cleanly!
        bi = numpy.array(range(self.nboards))
        self.iboard_by_face_board = bi.reshape(self.p.nfaces, self.p.face.nboards)
        self.iboard_by_conn_slot = bi.reshape(self.p.daq.nconnperwib, self.p.daq.nwibs)

        # List of indices to chips, accessed by [face,board_in_face,chip_on_board]
        ci = numpy.array(range(self.nchips))
        self.ichip_by_face_board_chip = ci.reshape(self.p.nfaces, self.p.face.nboards, self.p.board.nchips)


        # List of indices to  all conductors (or all channels) in an APA
        # accessed by [face, board_in_face, chip_in_board, chan_in_chip]
        ci = numpy.array(range(self.nchans))
        self.iconductor_by_face_board_chip_chan = ci.reshape(self.p.nfaces, self.p.face.nboards,
                                                             self.p.board.nchips, self.p.board.nchanperchip)

        # Flattened chip-channel to layer-conductor map
        self.ccls = flatten_cclsm()

    def iconductor_chip_chan(self, face, board_in_face, layer_in_face, wire_spot_in_layer):
        '''
        Given the paramers return information about the associated
        conductor as a triple:

            - iconductor :: the apa-global index for the conductor

            - chip :: the board-local index for the chip

            - chan :: the chip-local index for the channel
        '''
        # must +1 the spot to match the matrix assumption
        nchip,nch = self.ccls[("uvw"[layer_in_face], wire_spot_in_layer+1)]
        # must -1 the returns to match our assumption
        ichip,ich = nchip-1, nch-1
        icond = self.iconductor_by_face_board_chip_chan[face, board_in_face, ichip, ich]
        return (icond, ichip, ich)

    def iface_board(self, iboard):
        '''
        Given a global board index, return tuple of:

        - iface :: the apa-global face index
        - board : the face-local board index
        '''
        if not iboard in range(self.nboards):
            raise ValueError("iboard is out of range: %d" % iboard)
        iface = iboard//self.p.face.nboards
        board = iboard%self.p.face.nboards
        return (iface,board)


def graph(ac, makers):
    '''
    Return a directed graph expressing the connectivity and
    containment given the ApaConnectivity object.
    '''
    G = nx.DiGraph()
    
        

# The (#layers, #columns, #rows) for DUNE "anodes" in different detectors
oneapa_lcr = (1,1,1)
protodun_lcr = (1,2,3)
dune_lcr = (2,25,3)

