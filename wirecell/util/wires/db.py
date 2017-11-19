#!/usr/bin/env python

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
                          order_by='DetectorCrateLink.address',
                          secondary='detector_crate_links')
    crate_links = relationship('DetectorCrateLink',
                                order_by='DetectorCrateLink.address',
                                back_populates="detector")

    anodes = relationship('Anode',
                          secondary='detector_anode_links')
    anode_links = relationship('DetectorAnodeLink',
                               back_populates = "detector")

    def __repr__(self):
        return "<Detector: %s>" % self.id

class Anode(Base):
    '''
    An anode (APA, anode plane assembly) is in a detector and has two faces.
    '''
    __tablename__ = 'anodes'
    id = Column(Integer, primary_key=True)
    
    detectors = relationship('Detector',
                             secondary='detector_anode_links')
    detector_links = relationship('DetectorAnodeLink',
                                   back_populates = "anode")

    faces = relationship('Face',
                         secondary = 'anode_face_links')
    face_links = relationship('AnodeFaceLink',
                              back_populates = 'anode')

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
                               back_populates = 'face')

    planes = relationship('Plane',
                          secondary = 'face_plane_links')
    plane_links = relationship('FacePlaneLink',
                               back_populates = 'face')

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

    wies = relationship('Wire',
                        secondary = 'plane_wire_links')
    wire_links = relationship('PlaneWireLink',
                              back_populates = 'plane')

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
                        order_by = 'CrateWibLink.slot',
                        secondary = 'crate_wib_links')
    wib_links = relationship('CrateWibLink',
                             order_by = 'CrateWibLink.slot',
                             back_populates = 'crate')
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
                          order_by = 'WibBoardLink.connector',
                          secondary = 'wib_board_links')
    board_links = relationship('WibBoardLink',
                               order_by = 'WibBoardLink.connector',
                               back_populates = 'wib')
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
                                   back_populates = "board")
    chips = relationship("Chip",
                         secondary = "board_chip_links")
    chip_links = relationship("BoardChipLink",
                              back_populates = "board")
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
                                 back_populates = "chip")

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
                              back_populates = "conductor")


    channel_id = Column(Integer, ForeignKey('channels.id'))
    channel = relationship('Channel', back_populates='conductor')

    # ... add relationship to wires

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

    plane = Column(Integer)
    spot = Column(Integer)

    def __repr__(self):
        return "<BoardConductorLink plane:%s spot:%s [%s %s]> " % \
            (self.plane, self.spot, self.board, self.conductor)

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

    apanum = Column(Integer)

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

    # +/- 1
    side = Column(Integer)
    #fixme: constrain it to be u/v/w

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

    letter = Column(String(1))

    def __repr__(self):
        return "<FacePlaneLink letter:%s [%s %s]> " % \
            (self.letter, self.anode, self.face)
    
class PlaneWireLink(Base):
    '''
    A join table for plane-wire association
    '''
    __tablename__ = 'plane_wire_links'

    plane_id = Column(Integer, ForeignKey('planes.id'), primary_key = True)
    wire_id = Column(Integer, ForeignKey('wires.id'), primary_key = True)

    plane = relationship('Plane', back_populates='wire_links')
    wire = relationship('Wire', back_populates='plane_links')

    # wire in plane index
    index = Column(Integer)

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
