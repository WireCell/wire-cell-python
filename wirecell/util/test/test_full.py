#!/usr/bin/env python

from wirecell.util.wires import db
import numpy

# NOTE: the local wire attachement numbers count from 1.
chip_conductor_matrix = numpy.array([
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
 
def flatten_chip_conductor_map(ccm = chip_conductor_matrix):
    '''
    Flatten an ASIC channel X number matrix to a dictionary keyed by
    (plane letter, local wire attachment number (1-48 or 1-40).  Value
    is a tuple (ichip, ich) with ichip:{1-8} and ich:{1-16}
    '''
    ret = dict()
    for ichip, row in enumerate(ccm):
        for ich, cell in enumerate(row):
            cell = tuple(cell)
            ret[cell] = (ichip+1, ich+1)
    return ret
chip_conductor_map = flatten_chip_conductor_map()

def test_full():
    '''
    Test creating a full detector connectivity (no geometry)
    '''

    ses = db.session("sqlite:///test_full.db")

    # do we use one-based or zero-based counting:

    offset = 1

    # layers x columsn x rows
    oneapa_dim = (1,1,1)
    protodun_dim = (1,2,3)
    dune_dim = (2,25,3)
    apa_dim = dune_dim
    napas = reduce(lambda x,y: x*y, apa_dim)
    apa_indices = numpy.array(range(napas)).reshape(*apa_dim)

    def anode_crate_address(layer, column, row):
        'ad-hoc flattenning'
        return layer*1000+row*100+column

    # per apa parameters:

    nface_layers = 3

    nfaces = 2
    nface_spots = 10

    ncrate_wibs = 5
    nwib_conns = 4
    
    nchan_in_chip = 16
    nchip_on_board = 8

    nconductors_in_board_by_layer = (40,40,48)
    nconductors_in_board = sum (nconductors_in_board_by_layer)

    nboards = nfaces * nface_spots
    nchips = nboards*nchip_on_board
    nconductors = nboards * nconductors_in_board

    # need to locate a board by its face+spot and wib+conn
    # map (face,spot) to a board index 
    iboard_by_face_spot = numpy.array(range(nboards)).reshape(nfaces, nface_spots)
    # map (conn,wib) to a board index
    iboard_by_conn_wib = numpy.array(range(nboards)).reshape(nwib_conns, ncrate_wibs)

    ichip_by_face_board_chip = numpy.array(range(nchips))\
                                    .reshape(nfaces, nface_spots, nchip_on_board)

    # organize conductors indices into:
    # (2 faces x 10 boards x 16 ch x 8 chips)
    conductor_indices = numpy.asarray(range(nconductors))\
                             .reshape(nfaces, nface_spots,
                                      nchan_in_chip, nchip_on_board)
    def conductor_spot_map(face, board, layer, spot):
        '''
        map face, board in face, layer in board and spot in layer to a conductor index
        '''
        nchip,nch = chip_conductor_map[("uvw"[layer], spot+1)]
        ichip = nchip-1
        ich = nch-1
        icond = conductor_indices[face, board, ich, ichip]
        return (icond, ichip, ich)

    # now make

    det = db.Detector()

    seen_chips = set()

    for apa_layer in range(apa_dim[0]):
        for apa_column in range(apa_dim[1]):
            for apa_row in range(apa_dim[2]):
                apa_lcr = (apa_layer, apa_column, apa_row)
                print apa_lcr
                anode = db.Anode()
                det.add_anode(anode, *apa_dim)
                crate = db.Crate()
                det.add_crate(crate, anode_crate_address(*apa_lcr))

                wibs = [db.Wib() for n in range(ncrate_wibs)]
                faces = [db.Face() for n in range(nfaces)]
                planes = [db.Plane() for n in range(nfaces*nface_layers)]
                boards = [db.Board() for n in range(nboards)]
                chips = [db.Chip() for n in range(nchips)]
                conductors = [db.Conductor() for n in range(nconductors)]
                channels = [db.Channel() for n in range(nconductors)]

                for islot in range(ncrate_wibs):
                    wib = wibs[islot]
                    crate.add_wib(wib, islot+offset)
                    for iconnector in range(nwib_conns):
                        iboard = iboard_by_conn_wib[iconnector, islot]
                        board = boards[iboard]
                        wib.add_board(board, iconnector+offset)
                        iface = iboard//nface_spots
                        iboard_in_face = iboard%nface_spots
                        print '\t',iface,islot,iconnector,iboard,iboard_in_face
                        for ilayer, ispots in enumerate(nconductors_in_board_by_layer):
                            for ispot in range(ispots): # 40,40,48
                                icond,ichip,ich = conductor_spot_map(iface, iboard_in_face,
                                                                     ilayer, ispot)
                                conductor = conductors[icond]
                                board.add_conductor(conductor, ispot+offset, ilayer+offset)
                                
                                ichip_global = ichip_by_face_board_chip[iface,iboard_in_face,ichip]
                                chip = chips[ichip_global]
                                if not ichip_global in seen_chips:
                                    board.add_chip(chip, ichip+offset)
                                    seen_chips.add(ichip_global)
                                channel = channels[icond]
                                channel.conductor = conductor
                                chip.add_channel(channel, ich+offset)
                                # to do next: wires
    ses.add(det)
    ses.commit()
    
