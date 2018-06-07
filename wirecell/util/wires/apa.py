#!/usr/bin/env python
'''
This module provides connectivity information about DUNE APAs
(protoDUNE or nominal DUNE FD design, and not 35t prototype)

See the Plex class for a convenient one-stop user shop. 

'''
from wirecell import units
import wirecell.util.wires.generator as generator

from collections import namedtuple, Counter, defaultdict
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

# split off each half of the tuple.
chip_channel_spot = chip_channel_layer_spot_matrix[:,:,1].astype(numpy.int32)
chip_channel_layer = numpy.asarray(["uvw".index(i) for i in chip_channel_layer_spot_matrix[:,:,0].flat]).reshape(chip_channel_layer_spot_matrix.shape[:2]).astype(numpy.int32)


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


FaceParams = namedtuple("FaceParams", ["nlayers", "nboards"]);
BoardParams = namedtuple("BoardParams", ["nchips", "nchanperchip"])
DaqParams = namedtuple("DaqParams", ["nwibs", "nconnperwib"])
GeomParams = namedtuple("GeomParams", ["width", "height", "pitch","angle","offset","planex"])
Params = namedtuple("ApaParams", ["nfaces", "anode_loc", "crate_addr",
                                  "face", "board", "daq", "geom"])

plane_separation = 4.71*units.mm
default_params = Params(
    nfaces = 2,
    anode_loc = (1,1,1),        # layer, column, row
    crate_addr = 10101,          # arbitrary, take: layer*10000+column*100+row
    face = FaceParams(3, 10),
    board = BoardParams(8, 16),
    daq = DaqParams(5, 4),
    geom =  [
        GeomParams(
            width = 2295*units.mm,
            height = 5920*units.mm,
            pitch = 4.669*units.mm,
            angle = +35.707*units.deg,
            offset = 0.3923*units.mm, 
            planex = 3*plane_separation
        ),
        GeomParams(
            width = 2295*units.mm,
            height = 5920*units.mm,
            pitch = 4.669*units.mm,
            angle = -35.707*units.deg,
            offset = 0.3923*units.mm, 
            planex = 2*plane_separation
        ),
        GeomParams(
            width = 2295*units.mm,
            height = 5920*units.mm,
            pitch = 4.790*units.mm, 
            angle = 0.0,
            offset = 0.295*units.mm,
            planex = plane_separation
        ),
    ]
)

GeomPoint = namedtuple("GeomPoint", ["x","y","z"]) # face-specific coordinates
GeomWire = namedtuple("GeomWire", ["ploc", "wip", "spot", "seg", "p1" ,"p2"])

class Description(object):
    '''
    Provide data methods to describe an APA and enumerate its connectivity.
    '''
    def __init__(self,  params = default_params):
        self.p = params

        # Total numbers of things in one APA.
        # Some of this is just a copy in order to present a flat namespace
        self.nfaces = self.p.nfaces
        self.nplanes = self.nfaces * self.p.face.nlayers
        self.nwibs = self.p.daq.nwibs
        self.nboards = self.p.face.nboards*self.nfaces
        self.nchips = self.nboards * self.p.board.nchips
        self.nchannels = self.nchips*self.p.board.nchanperchip
        self.nconductors = self.nchannels
        #self.nwires, see below 

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
        # nominal: 2x10x8x16
        ci = numpy.array(range(self.nchannels))
        self.iconductor_by_face_board_chip_chan = ci.reshape(self.p.nfaces, self.p.face.nboards,
                                                             self.p.board.nchips, self.p.board.nchanperchip)

        # Flattened (layer-conductor)->(chip,channel) dictionary
        self.ccls = flatten_cclsm()
        counter = Counter()
        for k,v in self.ccls:
            counter[k[0]] += 1
        # nominal: (40,40,48)
        self.nch_in_board_by_layer = tuple([kv[1] for kv in sorted(counter.items())])


        self.points = list()

        ### wires ###
        #
        # Caution: wires are a bit tricky.  While each wire is
        # physically unique, each also shares a conceptual twin on the
        # other side of the APA, given rotational symmetry about the Y
        # axis.  These twins share the same points but these points
        # are expressed in two different coordinate systems, one for
        # each face!  When drawing a 2D representation of wires using
        # these points a rotation must be taken into account.
        self.wires_by_face_plane = [list(), list()]
        for iplane, geom in enumerate(self.p.geom):
            rect = generator.Rectangle(geom.width, geom.height)
            # (ap, side, spot, seg, p1, p2)
            raw_wires = generator.wrapped_from_top_oneside(geom.offset, geom.angle, geom.pitch, rect)
            raw_wires.sort()
            gwires_front = list()
            gwires_back = list()
            for wip, raw_wire in enumerate(raw_wires):
                ap, side, spot, seg, zy1, zy2 = raw_wire
                ip1 = len(self.points)
                p1 = GeomPoint(geom.planex, zy1[1], zy1[0])
                self.points.append(p1)
                ip2 = len(self.points)
                p2 = GeomPoint(geom.planex, zy2[1], zy2[0])
                self.points.append(p2)
                wf = GeomWire(ap, wip, spot, seg, ip1, ip2)
                wb = GeomWire(ap, wip, spot, seg, ip1, ip2)
                gwires_front.append(wf)
                gwires_back.append(wb)
            self.wires_by_face_plane[0].append(gwires_front)
            self.wires_by_face_plane[1].append(gwires_back)

        self.nwires_by_plane = [len(l) for l in self.wires_by_face_plane[0]]
        self.nwires_per_face = sum(self.nwires_by_plane)
        self.nwires = self.nfaces * self.nwires_per_face
        self.npoints = len(self.points)

    def wire_index_by_wip(self, face, plane, wip):
        '''
        Return a tuple of a global wire index and the gwire
        '''
        index = face*self.nwires_per_face + sum(self.nwires_by_plane[:plane]) + wip
        wire = self.wires_by_face_plane[face][plane][wip]
        return (index,wire)

    def iconductor_by_face_plane_spot(self, face, plane_in_face, spot_in_plane):
        '''
        Return the global conductor index based on the face, plane and spot.
        '''
        nch = self.nch_in_board_by_layer[plane_in_face] # 40,40,48
        board_in_face = spot_in_plane//nch
        spot_in_layer = spot_in_plane%nch
        return self.iconductor_chip_chan(face, board_in_face, plane_in_face, spot_in_layer)[0]


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

    def iplane(self, iface, plane_in_face):
        'Return global plane index given global face and plane in face'
        # trivial...
        return iface*self.p.face.nlayers + plane_in_face

    

    pass


# A set of makers to create objects in the APA heirachy.  Each call to
# a maker should return an instance of its associated type.  This
# construction call should take parameters with names taken from by
# its connected types which should accept values as returned by other
# makers.  All construction arguments are optional and may be set as
# attributes after an object is created.  If an object has a
# one-to-many relationship with another object then the paramter name
# is pluralized (in all case simply by appending "s"), else it is
# singular.

Parts = namedtuple("Parts",
                   ["detector", "apa", "face", "plane",
                    "wib", "board", "chip", "conductor",
                    "channel", "wire", "point"
                   ])
    
def maker(G, ac, typename):
    try:
        n = getattr(ac, 'n'+typename+'s')
    except AttributeError:
        G.add_node(typename, type=typename)
        return typename
    ret = list()
    for ind in range(n):
        name = "%s%d" % (typename, ind)
        G.add_node(name, type=typename, index=ind)
        ret.append(name)
    return ret


def graph(desc, maker = maker):
    '''
    Return a directed graph expressing the connectivity and
    containment given the apa.Description object.
    '''
    G = networkx.DiGraph()
    
    p = Parts(*[maker(G, desc, typename) for typename in Parts._fields])

    def join(n1, n2, link, relationship = "containment", **params):

        if relationship in ["peer", "sibling"]:
            G.add_edge(n1, n2, relationship=relationship, link=link, **params)
            G.add_edge(n2, n1, relationship=relationship, link=link, **params)            
            return
        if relationship in ["direct","simple"]:
            G.add_edge(n1, n2, relationship=relationship, link=link, **params)
            return
        # parent/child
        G.add_edge(n1, n2, relationship="children", link=link, **params)
        G.add_edge(n2, n1, relationship="parent", link=link, **params)
        
    # FIXME: for now hard-code the single apa connection.  This should
    # be expanded to 6 for protoDUNE or 150 for DUNE SP FD.  But, each
    # APA is identical up to a translation in global coordinates so it
    # is not yet obvious that it's necessary to exaustively construct
    # them all.
    join(p.detector, p.apa, 'submodule', anode_lcr = desc.p.anode_loc)

    # variable name convention below: gi_ = "global index", "i_" just an index.
    # everything else an object
    for gi_wib in range(desc.nwibs):
        wib = p.wib[gi_wib]

        join(p.apa, wib, 'slot', slot=gi_wib)


        for i_wibconn in range(desc.p.daq.nconnperwib):
            gi_board = desc.iboard_by_conn_slot[i_wibconn, gi_wib]
            board = p.board[gi_board]

            join(wib, board, 'cable', connector = i_wibconn)

            gi_face, iboard_in_face = desc.iface_board(gi_board)

            for ilayer_in_face, ispots_in_layer in enumerate(desc.nch_in_board_by_layer):
                for ispot_in_layer in range(ispots_in_layer): # 40,40,48
                    gi_cond, ichip_on_board, ichan_in_chip \
                        = desc.iconductor_chip_chan(gi_face, iboard_in_face,
                                                  ilayer_in_face, ispot_in_layer)
                                            
                    conductor = p.conductor[gi_cond]
                    join(board, conductor, 'trace', layer=ilayer_in_face, spot=ispot_in_layer)

                    gi_chip = desc.ichip_by_face_board_chip[gi_face, iboard_in_face, ichip_on_board]
                    chip = p.chip[gi_chip]

                    # Note: this will be called multiple times.  Since
                    # G is not a multigraph subsequent calls are no-ops
                    join(board, chip, 'place', spot=ichip_on_board)

                    # channels and conductors are one-to-one
                    channel = p.channel[gi_cond]
                    join(chip, channel, 'address', address=ichan_in_chip)

                    join(channel, conductor, 'channel', relationship="peer")

    for ipoint, point in enumerate(p.point):
        G.node[point]['pos'] = desc.points[ipoint]

    for gi_face in range(desc.nfaces):
        face = p.face[gi_face]
        join(p.apa, face, 'side', side=gi_face)

        for iboard_in_face in range(desc.p.face.nboards):
            gi_board = desc.iboard_by_face_board[gi_face, iboard_in_face]
            board = p.board[gi_board]
            join(face, board, 'spot', spot = iboard_in_face)

        for iplane_in_face, wires in enumerate(desc.wires_by_face_plane[gi_face]):

            gi_plane = desc.iplane(gi_face, iplane_in_face)
            plane = p.plane[gi_plane]
            join(face, plane, 'plane', plane=iplane_in_face)

            for wip in range(desc.nwires_by_plane[iplane_in_face]):
                gi_wire, gwire = desc.wire_index_by_wip(gi_face, iplane_in_face, wip)
                wire = p.wire[gi_wire]
                join(plane, wire, 'wip', wip=wip)
                G.node[wire]['pitchloc'] = gwire.ploc

                # odd segments are on the "other" face.
                spot_face = (gi_face + gwire.seg%2)%2

                gi_conductor = desc.iconductor_by_face_plane_spot(spot_face, iplane_in_face, gwire.spot)
                conductor = p.conductor[gi_conductor]
                join(conductor, wire, 'segment', segment=gwire.seg)

                tail = p.point[gwire.p1]
                join(wire, tail, 'pt', endpoint=0)
                head = p.point[gwire.p2]
                join(wire, head, 'pt', endpoint=1)

    return G,p

# The (#layers, #columns, #rows) for DUNE "anodes" in different detectors
oneapa_lcr = (1,1,1)
protodun_lcr = (1,2,3)
dune_lcr = (2,25,3)


def channel_tuple(G, wire):
    '''
    Return a channel address tuple associated with the given wire.

    The tuple is intended to be directly passed to channel_hash().
    '''

    # fixme: some problems with dependencies here:
    from .graph import parent
    
    conductor = parent(G, wire, 'conductor')
    channel = parent(G, conductor, 'channel')
    chip = parent(G, channel, 'chip')
    board = parent(G, chip, 'board')
    box = parent(G, board, 'face')
    wib = parent(G, board, 'wib')
    apa = parent(G, wib, 'apa')
    
    islot = G[apa][wib]['slot']
    iconn = G[wib][board]['connector']
    ichip = G[board][chip]['spot']
    iaddr = G[chip][channel]['address']
    return (iconn, islot, ichip, iaddr)

def channel_hash(iconn, islot, ichip, iaddr):
    '''
    Hash a channel address tuple into a single integer.  

    See also channel_tuple().
    '''
    return int("%d%d%d%02d" % (iconn+1, islot+1, ichip+1, iaddr+1))

def channel_unhash(chident):
    '''
    Return tuple used to produce the given hash.
    '''
    a = str(chident)
    return tuple([int(n)-1 for n in [a[0], a[1], a[2], a[3:5]]])


def channel_ident(G, wire):
    '''
    Return an identifier number for the channel attached to the given wire.
    '''
    return channel_hash(*channel_tuple(G, wire))



class Plex(object):
    '''
    Provide an instance of a Plex that provides some convenient
    mappings on the connection graph.

    This assumes the given graph was created using conventions
    expressed in parent module.

    The term "Plex" for this role was invented by MINOS.
    '''

    def __init__(self, G, P):
        self.G = G
        self.P = P

    def channel_plane(self, chanidents):
        '''
        Return a generator of plane numbers, one for each chanidents.
        '''
        for chid in chanidents:
            iconn, islot, ichip, iaddr = channel_unhash(chid)
            yield chip_channel_layer[ichip, iaddr]
            
