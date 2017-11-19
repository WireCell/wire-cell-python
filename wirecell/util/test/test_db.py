#!/usr/bin/env python
import wirecell.util.wires.db as db

def test_classes():
    print
    ses = db.session()

    # parent
    det = db.Detector()
    cr1 = db.Crate()
    cr2 = db.Crate()

    dce1 = db.DetectorCrateLink(detector=det, crate=cr1, address=42)
    dce2 = db.DetectorCrateLink(detector=det, crate=cr2, address=40)
    # Note: there is a temptation to add a crate directly to the
    # .crate relationship.  This doesn't sync the link. so no address
    # can ever be set.  A Detector.add_crate(crate, address) could
    # help.  But, whatever, don't do the following:
    # 
    #det.crates.append(cr2)

    wibs = list()
    for slot in range(5):
        wib = db.Wib()
        wibs.append(wib)
        db.CrateWibLink(crate=cr1, wib=wib, slot=slot)

    boards = list()
    for connector in range(4):
        board = db.Board()
        boards.append(board)
        db.WibBoardLink(wib=wibs[0], board=board, connector=connector)

    conductors = list()
    for plane,spots in enumerate([40,40,48]):
        for spot in range(spots):
            conductor = db.Conductor()
            conductors.append(conductor)
            db.BoardConductorLink(board=boards[0], conductor=conductor,
                                  plane=plane, spot=spot)
    chips = list();
    for spot in range(8):
        chip = db.Chip()
        chips.append(chip)
        db.BoardChipLink(board=boards[0], chip=chip, spot=spot)

    chans = list()
    for address in range(16):
        # note: this is a totally bogus channel map! just for testing.
        chan = db.Channel(conductor=conductors[address])
        chans.append(chan)
        db.ChipChannelLink(chip=chips[0], channel=chan, address=address)


    ses.add(det)

    ses.commit()

    det = ses.query(db.Detector).one()

    print 'DET:',det
    print 'DET.crates:', det.crates
    print 'DET.crate_links:', det.crate_links
    print 'DET.crates[0].detectors:', det.crates[0].detectors
    print 'DET.crates[1].detectors:', det.crates[1].detectors

    assert det.crates[0].detectors[0] == det
    assert det.crate_links[0].address == 40
    assert det.crates[0] == det.crate_links[0].crate

    crate42 = ses.query(db.DetectorCrateLink).\
              filter(db.DetectorCrateLink.address==42).one().crate
    print crate42
    assert crate42 == det.crate_links[1].crate

    print crate42.wibs[0]
    board = crate42.wibs[0].boards[0]
    print board

    print board.conductors
    print board.chips
    chip = board.chips[0]
    print chip.channels
    for ch in chip.channels:
        print ch, ch.conductor
    return

