#!/usr/bin/env python
'''
Organize the connectivity of the parts of protoDUNE and DUNE FD
single-phase detectors.

FIXME: only TPC related parts are supported for now!

FIXME: this design is currently ambigous as to whether each part is
"logical" or "touchable" (in the Geant4 sense of these words).  It
allows for both and it is up to the user to determine interpretation.
Future development may provide explicit interpretation.

This organization separates into the following conceptual parts:

    - part identifiers :: the database provides a table for each type
      of detector part relevant to identifiers that may be present in
      its data.

    - part connectivity :: relationships between part tables and join
      tables form edges in an overall connectivity graph.

    - connectivity ordering :: edges in the connectivity graph have a
      scalar ordering based on one or more table columns or class
      attributes.

    - numbering convention :: the database does not assume any
      particular numbering convention except in so much as it can be
      expressed in terms of the connectivity ordering attributes.

A part table has relationships to others which fall into one of these
categories:

    - child :: a part may have a child sub-part which is considered
      contained by the part.

    - parent :: a part may itself be a child and maintain a connection
      to its parent part.

    - peer :: two parts may reference each other

    - simple :: a table may simply reference another in a
      unidirectional manner.

    - special :: a relationship of convenience

The child and parent connectivity is maintened, in part, through a
join table ("link") which allows specifying one or more connectivity
ordering attributes.  These conections are expressed in two
relationships as rows/attributes on the part table/class.  they take
names after the "other" part:

    - parts :: a sequence of the other connected parts with has no
      explicit order.

    - part_links :: a sequence of links from the part to its children
      or its parent(s).  When referencing children, the links are
      ordered by the connectivit ordering attributes held in the link
      objects.  Each link object gives access to both parent and child
      parts.

As a convenience, a child part may be associated to a parent part
through the parent's `add_<part>()` method.  These methods accept an
existing child object and all connection ordering parameters.  A join
object is constructed and returned (but can usually be ignored).

Note, in detector description limited to a single detector where each
part is considered "touchable" (see FIXME above) the parent-child
relationship is one-to-many.  OTOH, multiple connections from one
child back to multiple parents are supported.  As an example, a
database may contain descriptions for both protoDUNE and DUNE where
with two Detector objects and where the former merely references six
anodes of the latter.

The branches of the parent-child tree terminate at the channels leaf,
although channels have "peer" links to conductors (see below).
Conceptually, if one considers a containment tree, the tree also
terminates at wire leafs.  However, there are cycles in relationship.
Wires are contained both by conductors which connect up through
electronics back to the detector and in planes which connect up
through structural elements to the detector.  Likewise boards are
conceptually "in" both WIBs and Faces.

The "peer" relationship is a simple one-to-one connection.  Each part
on either end of the connection has an attribute named after the
"other" part.  These are used to directly express a connection which
either can not be expressed or which would require traversal up and
down many intermeidate branches of the parent/child tree.  For example
a Channel has a "conductor" and a Conductor has a "channel".  Also,
Crates and Anodes share a peer relationship.  To indirectly associate
the two would require redundancy in the join tables and require
complex queries.  Instead, an explicit relationship is formed which
represents a physical soldering of a conductor to a wire board layer
and spot.  A side effect benefit is that mistakes in such physical
connections can be represented.

The "peer" relationships must be set explicitly set either at the time
of constructing the final end point object or later by setting the
corresponsing attribute on one (not both) of the end point objects.

A "simple" relationship is used to compose compound values such as
referencing rays which themselves reference points.  In some cases
"special" relationships are provided.  For example, a Point can
determine which Rays refer to it, but other tables may reference a
Point and no back reference will exist.
'''



from sqlalchemy import Column, ForeignKey, Float, Integer, String, CheckConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()
metadata = Base.metadata

class Detector(Base):
    '''
    One detector, eg protoDUNE/SP, DUNE FD SP module.
    '''
    __tablename__ = "detectors"
    id = Column(Integer, primary_key=True)
    
    crates = relationship('Crate',
                          secondary='detector_crate_links')
    crate_links = relationship('DetectorCrateLink',
                          order_by='DetectorCrateLink.address',
                                back_populates="detector")

    anodes = relationship('Anode',
                          secondary='detector_anode_links')
    anode_links = relationship('DetectorAnodeLink',
                               order_by='DetectorAnodeLink.layer, '
                               'DetectorAnodeLink.column, '
                               'DetectorAnodeLink.row',
                               back_populates = "detector")

    def add_anode(self, anode, row, column, layer):
        '''
        Add an anode at a given address, return the DetectorAnodeLink.
        '''
        return DetectorAnodeLink(detector=self, anode=anode,
                                 row=row, column=column, layer=layer)

    def add_crate(self, cr, address):
        '''
        Add a create at a given address, return the DetectorCrateLink.
        '''
        return DetectorCrateLink(detector=self, crate=cr, address=address)

    def __repr__(self):
        return "<Detector: %s>" % self.id

class Anode(Base):
    '''
    An anode (APA, anode plane assembly) is in a detector and has two faces.
    '''
    __tablename__ = 'anodes'
    id = Column(Integer, primary_key=True)
    
    # Note, this is in general a collection.  For example, a user may
    # define protoDUNE as a subset of DUNE anodes in which case some
    # anodes may have two detectors.
    detectors = relationship('Detector',
                             secondary='detector_anode_links')
    detector_links = relationship('DetectorAnodeLink',
                                   back_populates = "anode")

    faces = relationship('Face',
                         secondary = 'anode_face_links')
    face_links = relationship('AnodeFaceLink',
                              order_by='AnodeFaceLink.side',
                              back_populates = 'anode')

    def add_face(self, face, side):
        '''
        Add a face to this anode on the given side.
        '''
        return AnodeFaceLink(anode=self, face=face, side=side)

    # origin_id = Column(Integer, ForeignKey('points.id'))
    # ... link directly to crate
    def __repr__(self):
        return "<Anode: %s>" % self.id


class Face(Base):
    '''
    An anode face contains boards and planes.
    '''
    __tablename__ = 'faces'
    id = Column(Integer, primary_key=True)
    
    anodes = relationship('Anode',
                          secondary='anode_face_links')
    anode_links = relationship('AnodeFaceLink',
                               back_populates = "face")

    boards = relationship('Board',
                          secondary = 'face_board_links')
    board_links = relationship('FaceBoardLink',
                               order_by = 'FaceBoardLink.spot',
                               back_populates = 'face')

    planes = relationship('Plane',
                          secondary = 'face_plane_links')
    plane_links = relationship('FacePlaneLink',
                               order_by = 'FacePlaneLink.layer',
                               back_populates = 'face')

    def add_board(self, board, spot):
        '''
        Add a board at a spot into this face.
        '''
        return FaceBoardLink(face=self, board=board, spot=spot)

    def add_plane(self, plane, layer):
        '''
        Add a plane at a layer into this face.
        '''
        return FacePlaneLink(face=self, board=board, layer=layer)

    # ...drift dir
    def __repr__(self):
        return "<Face: %s>" % self.id


class Plane(Base):
    '''
    A plane is in a face and has wires.
    '''
    __tablename__ = 'planes'
    id = Column(Integer, primary_key=True)
    
    faces = relationship('Face',
                         secondary='face_plane_links')
    face_links = relationship('FacePlaneLink',
                              back_populates = "plane")

    wires = relationship('Wire',
                         secondary = 'plane_wire_links')
    wire_links = relationship('PlaneWireLink',
                              order_by = 'PlaneWireLink.spot',
                              back_populates = 'plane')

    def add_wire(self, wire, spot):
        '''
        Add a wire in a particular spot in the plane.
        '''
        return PlaneWireLink(plane=self, wire=wire, spot=spot)

    #... wiredir, pitchdir, origin

    def __repr__(self):
        return "<Plane: %s>" % self.id

class Crate(Base):
    '''
    One crate of a detector holding WIBs.
    '''
    __tablename__ = "crates"
    id = Column(Integer, primary_key=True)

    detectors = relationship('Detector',
                             secondary='detector_crate_links')
    detector_links = relationship('DetectorCrateLink',
                                   back_populates = "crate")

    wibs = relationship('Wib',
                        secondary = 'crate_wib_links')
    wib_links = relationship('CrateWibLink',
                             order_by = 'CrateWibLink.slot',
                             back_populates = 'crate')

    def add_wib(self, wib, slot):
        '''
        Add a WIB into the slot of this crate.
        '''
        return CrateWibLink(crate=self, wib=wib, slot=slot)

    def __repr__(self):
        return "<Crate: %s>" % self.id

    
class Wib(Base):
    '''
    A WIB (warm interface board) sits in a crate and connects to a
    four (electronics wire) boards.
    '''
    __tablename__ = "wibs"
    id = Column(Integer, primary_key=True)

    crates = relationship("Crate",
                          secondary = 'crate_wib_links')
    crate_links = relationship('CrateWibLink',
                               back_populates = 'wib')

    boards = relationship('Board',
                          secondary = 'wib_board_links')
    board_links = relationship('WibBoardLink',
                               order_by = 'WibBoardLink.connector',
                               back_populates = 'wib')
    def add_board(self, board, connector):
        '''
        Add an cold electronics wire board to this WIB
        '''
        return WibBoardLink(wib=self, board=board, connector=connector)

    def __repr__(self):
        return "<Wib: %s>" % self.id


class Board(Base):
    '''
    An (electronics wire) board sits on top (or bottom) of an APA
    frame and holds 8 pairs of FE/ADC ASIC chips and connects to three
    rows of 40, 40 and 48 conductors.
    '''
    __tablename__ = "boards"
    id = Column(Integer, primary_key=True)

    # up links
    wibs = relationship("Wib",
                        secondary = "wib_board_links")
    wib_links = relationship("WibBoardLink",
                             back_populates = "board")

    faces = relationship("Face",
                        secondary = "face_board_links")
    face_links = relationship("FaceBoardLink",
                             back_populates = "board")

    # downlinks
    conductors = relationship("Conductor",
                              secondary = "board_conductor_links")
    conductor_links = relationship("BoardConductorLink",
                                   order_by = 'BoardConductorLink.layer, '
                                   'BoardConductorLink.spot',
                                   back_populates = "board")
    chips = relationship("Chip",
                         secondary = "board_chip_links")
    chip_links = relationship("BoardChipLink",
                              order_by = 'BoardChipLink.spot',
                              back_populates = "board")

    def add_conductor(self, conductor, spot, layer):
        '''
        Add a conductor to this board at the given spot of a layer.
        '''
        return BoardConductorLink(board=self, conductor=conductor, spot=spot, layer=layer)

    def add_chip(self, chip, spot):
        '''
        Add a chip to this board at the given spot.
        '''
        return BoardChipLink(board=self, chip=chip, spot=spot)

    def __repr__(self):
        return "<Board: %s>" % self.id
    

class Chip(Base):
    '''
    A chip is a pair of FE/ADC asics with 16 channels
    '''
    __tablename__ = "chips"
    id = Column(Integer, primary_key=True)

    boards = relationship("Board",
                          secondary = "board_chip_links")
    board_links = relationship("BoardChipLink",
                               back_populates = "chip")

    channels = relationship("Channel",
                            secondary = "chip_channel_links")
    channel_links = relationship("ChipChannelLink",
                                 order_by = 'ChipChannelLink.address',
                                 back_populates = "chip")

    def add_channel(self, channel, address):
        '''
        Add a channel to this chip at the given address.
        '''
        return ChipChannelLink(chip=self, channel=channel, address=address)

    def __repr__(self):
        return "<Chip: %s>" % self.id
    
class Conductor(Base):
    '''
    A conductor is a length of wire segments connecting to a board at
    a given position indicated by plane and spot .
    '''
    __tablename__ = "conductors"
    id = Column(Integer, primary_key=True)

    boards = relationship("Board",
                          secondary = "board_conductor_links")
    board_links = relationship("BoardConductorLink",
                               back_populates = "conductor")

    wires = relationship("Wire",
                         secondary = "conductor_wire_links")
    wire_links = relationship("ConductorWireLink",
                              order_by = 'ConductorWireLink.segment',
                              back_populates = "conductor")

    channel_id = Column(Integer, ForeignKey('channels.id'))
    channel = relationship('Channel', back_populates='conductor')

    def add_wire(self, wire, segment):
        '''
        Add a wire to this conductor as the given segment.
        '''
        return ConductorWireLink(conductor=self, wire=wire, segment=segment);

    def __repr__(self):
        return "<Conductor: %s>" % self.id


class Channel(Base):
    '''
    An electronics channel.
    '''
    __tablename__ = 'channels'
    id = Column(Integer, primary_key=True)

    chips = relationship("Chip",
                          secondary = "chip_channel_links")
    chip_links = relationship("ChipChannelLink",
                               back_populates = "channel")

    # Direct relationship to one and only attached conductor 
    conductor = relationship('Conductor', uselist=False,
                             back_populates='channel')
    def __repr__(self):
        return "<Channel: %s>" % self.id
   

class Wire(Base):
    '''
    A wire segment.
    '''
    __tablename__ = 'wires'
    id = Column(Integer, primary_key=True)

    conductors = relationship("Conductor",
                              secondary = "conductor_wire_links")
    conductor_links = relationship("ConductorWireLink",
                                   back_populates = "wire")

    planes = relationship("Plane",
                          secondary = "plane_wire_links")
    plane_links = relationship("PlaneWireLink",
                               back_populates = "wire")


    ray_id = Column(Integer, ForeignKey('rays.id'))
    ray = relationship('Ray')

    def __repr__(self):
        return "<Wire %d>" % self.id


class Ray(Base):
    '''
    Two endpoints and a microphone.
    '''
    __tablename__ = "rays"

    id = Column(Integer, primary_key=True)

    # Endpoint tail point 
    tail_id = Column(Integer, ForeignKey('points.id'))
    tail = relationship("Point", foreign_keys = [tail_id])

    # Endpoint head point 
    head_id = Column(Integer, ForeignKey('points.id'))
    head = relationship("Point", foreign_keys = [head_id])

    # The wire segments which have this ray as their endpoints
    wires = relationship("Wire", 
                         primaryjoin='Ray.id == Wire.ray_id')
    def __repr__(self):
        return "<Ray %d>" % self.id


class Point(Base):
    '''
    A point in some unspecified 3-space coordinate system.
    '''
    __tablename__ = "points"

    id = Column(Integer, primary_key=True)
    x = Column(Float)
    y = Column(Float)
    z = Column(Float)
    
    # the ray tails using this point
    tails = relationship("Ray",
                         primaryjoin='Point.id == Ray.tail_id')
    # the ray heads using this point
    heads = relationship("Ray",
                         primaryjoin='Point.id == Ray.head_id')
    # the rays using this point (union of tails and heads)
    rays = relationship("Ray",
                        primaryjoin='or_(Point.id == Ray.head_id, Point.id == Ray.tail_id)')

    def __repr__(self):
        return "<Point %d x:%f y:%f z:%f>" % (self.id, self.x, self.y, self.z)



# Association pattern
# http://docs.sqlalchemy.org/en/latest/orm/basic_relationships.html#association-object

class DetectorCrateLink(Base):
    '''
    A join table for detector-crate association.
    '''
    __tablename__ = "detector_crate_links"
    
    detector_id = Column(Integer, ForeignKey('detectors.id'), primary_key=True)
    crate_id = Column(Integer, ForeignKey('crates.id'), primary_key=True)
    detector = relationship('Detector', back_populates='crate_links')
    crate = relationship('Crate', back_populates='detector_links')

    address = Column(Integer)

    def __repr__(self):
        return "<DetectorCrateLink address:%s [%s %s]> " % \
            (self.address, self.detector, self.crate)
    
class CrateWibLink(Base):
    '''
    A join table for crate-WIB association
    '''
    __tablename__ = "crate_wib_links"
    crate_id = Column(Integer, ForeignKey('crates.id'), primary_key=True)    
    wib_id = Column(Integer, ForeignKey('wibs.id'), primary_key=True)    

    crate = relationship('Crate', back_populates='wib_links')
    wib = relationship('Wib', back_populates='crate_links')

    slot = Column(Integer)

    def __repr__(self):
        return "<CrateWibLink slot:%s [%s %s]> " % \
            (self.address, self.crate, self.wib)

class WibBoardLink(Base):
    '''
    A join table for WIB-board association.
    '''
    __tablename__ = 'wib_board_links'

    wib_id = Column(Integer, ForeignKey('wibs.id'), primary_key = True)
    board_id = Column(Integer, ForeignKey('boards.id'), primary_key = True)

    wib = relationship('Wib', back_populates='board_links')
    board = relationship('Board', back_populates='wib_links')

    # wibs have their data connectors arranged on a 2x2 grid but are
    # identified in the design with a single dimensional connector
    # number and well defined so the identifier is kept scalar.
    connector = Column(Integer)

    def __repr__(self):
        return "<WibBoardLink slot:%s [%s %s]> " % \
            (self.connector, self.wib, self.board)


class BoardConductorLink(Base):
    '''
    A join table for board-conductor association.
    '''
    __tablename__ = 'board_conductor_links'

    board_id = Column(Integer, ForeignKey('boards.id'), primary_key = True)
    conductor_id = Column(Integer, ForeignKey('conductors.id'), primary_key = True)

    board = relationship('Board', back_populates='conductor_links')
    conductor = relationship('Conductor', back_populates='board_links')

    # A conductor is attached to a wire board in one of three possible
    # conceptual lines, one for each plane layer.  The number of
    # conductors is not the same for each layer.  The spot identifies
    # a conductor in its layer.
    layer = Column(Integer)
    spot = Column(Integer)

    def __repr__(self):
        return "<BoardConductorLink layer:%s spot:%s [%s %s]> " % \
            (self.plane, self.layer, self.board, self.conductor)

class BoardChipLink(Base):
    '''
    A join table for board-chip association.
    '''
    __tablename__ = 'board_chip_links'

    board_id = Column(Integer, ForeignKey('boards.id'), primary_key = True)
    chip_id = Column(Integer, ForeignKey('chips.id'), primary_key = True)

    board = relationship('Board', back_populates='chip_links')
    chip = relationship('Chip', back_populates='board_links')

    spot = Column(Integer)

    def __repr__(self):
        return "<BoardChipLink spot:%s [%s %s]> " % \
            (self.spot, self.board, self.chip)


class ChipChannelLink(Base):
    '''
    A join table for chip-channel association.
    '''
    __tablename__ = 'chip_channel_links'

    chip_id = Column(Integer, ForeignKey('chips.id'), primary_key = True)
    channel_id = Column(Integer, ForeignKey('channels.id'), primary_key = True)

    chip = relationship('Chip', back_populates='channel_links')
    channel = relationship('Channel', back_populates='chip_links')

    address = Column(Integer)

    def __repr__(self):
        return "<ChipOffsetLink address:%s [%s %s]> " % \
            (self.address, self.chip, self.channel)

class ConductorWireLink(Base):
    '''
    A join table for conductor-wire association.
    '''
    __tablename__ = 'conductor_wire_links'

    conductor_id = Column(Integer, ForeignKey('conductors.id'), primary_key = True)
    wire_id = Column(Integer, ForeignKey('wires.id'), primary_key = True)

    conductor = relationship('Conductor', back_populates='wire_links')
    wire = relationship('Wire', back_populates='conductor_links')

    segment = Column(Integer)

    def __repr__(self):
        return "<ConductorWireLink segment:%s [%s %s]> " % \
            (self.segment, self.conductor, self.wire)


class DetectorAnodeLink(Base):
    '''
    A join table for detector-anode association
    '''
    __tablename__ = 'detector_anode_links'

    detector_id = Column(Integer, ForeignKey('detectors.id'), primary_key = True)
    anode_id = Column(Integer, ForeignKey('anodes.id'), primary_key = True)

    detector = relationship('Detector', back_populates='anode_links')
    anode = relationship('Anode', back_populates='detector_links')

    # APAs with their edges touching form a row.  
    row = Column(Integer)
    # A column of APAs are formed along direction or anti-direction of the drift
    column = Column(Integer)
    # A layer of APAs are formed when one is stacked on top of another
    # along the top or bottom of their frames.  protoDUNE has only one
    # layer, DUNE has two.
    layer = Column(Integer)

    def __repr__(self):
        return "<DetectorAnodeLink apanum:%s [%s %s]> " % \
            (self.apanum, self.detecotr, self.anode)

class AnodeFaceLink(Base):
    '''
    A join table for anode-face association
    '''
    __tablename__ = 'anode_face_links'

    anode_id = Column(Integer, ForeignKey('anodes.id'), primary_key = True)
    face_id = Column(Integer, ForeignKey('faces.id'), primary_key = True)

    anode = relationship('Anode', back_populates='face_links')
    face = relationship('Face', back_populates='anode_links')

    # Enumerate on which side of the anode this face resides.
    side = Column(Integer)

    def __repr__(self):
        return "<AnodeFaceLink side:%s [%s %s]> " % \
            (self.side, self.anode, self.face)
    
class FaceBoardLink(Base):
    '''
    A join table for face-baord association
    '''
    __tablename__ = 'face_board_links'

    face_id = Column(Integer, ForeignKey('faces.id'), primary_key = True)
    board_id = Column(Integer, ForeignKey('boards.id'), primary_key = True)

    face = relationship('Face', back_populates='board_links')
    board = relationship('Board', back_populates='face_links')

    spot = Column(Integer)

    def __repr__(self):
        return "<FaceBoardLink spot:%s [%s %s]> " % \
            (self.spot, self.anode, self.face)
    
class FacePlaneLink(Base):
    '''
    A join table for face-plane association
    '''
    __tablename__ = 'face_plane_links'

    face_id = Column(Integer, ForeignKey('faces.id'), primary_key = True)
    plane_id = Column(Integer, ForeignKey('planes.id'), primary_key = True)

    face = relationship('Face', back_populates='plane_links')
    plane = relationship('Plane', back_populates='face_links')

    layer = Column(Integer)

    def __repr__(self):
        return "<FacePlaneLink layer:%s [%s %s]> " % \
            (self.layer, self.anode, self.face)
    
class PlaneWireLink(Base):
    '''
    A join table for plane-wire association
    '''
    __tablename__ = 'plane_wire_links'

    plane_id = Column(Integer, ForeignKey('planes.id'), primary_key = True)
    wire_id = Column(Integer, ForeignKey('wires.id'), primary_key = True)

    plane = relationship('Plane', back_populates='wire_links')
    wire = relationship('Wire', back_populates='plane_links')

    # wire in some spot of the plane.
    spot = Column(Integer)

    def __repr__(self):
        return "<PlaneWireLink index:%s [%s %s]> " % \
            (self.index, self.plane, self.wire)



from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
def session(dburl="sqlite:///:memory:"):
    '''
    Return a DB session
    '''
    engine = create_engine(dburl)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()
