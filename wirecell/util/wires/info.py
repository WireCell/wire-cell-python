#!/usr/bin/env python
'''
Functions to provide information about wires
'''
from wirecell import units
from .common import Ray
import numpy
import math

def p2p(p):
    return dict(x=p.x, y=p.y, z=p.z)

def todict(store):
    '''
    Convert the store to a dict-based representation.  This will
    inflate any duplicate references
    '''
    d_anodes = list()
    for anode in store.anodes:
        d_anode = dict(ident = anode.ident, faces = list())
        for iface in anode.faces:
            face = store.faces[iface]
            d_face = dict(ident = face.ident, planes = list())
            for iplane in face.planes:
                plane = store.planes[iplane]
                d_plane = dict(ident = plane.ident, wires = list())

                #print ("anode:%d face:%d plane:%d" % (anode.ident, face.ident, plane.ident))

                for wind in plane.wires:
                    wire = store.wires[wind]
                    d_wire = dict(ident = wire.ident,
                                  channel = wire.channel,
                                  segment = wire.segment,
                                  head = p2p(store.points[wire.head]),
                                  tail = p2p(store.points[wire.tail]))
                    d_plane["wires"].append(d_wire)
                d_face["planes"].append(d_plane)
            d_anode["faces"].append(d_face)
        d_anodes.append(d_anode)

    # fixme: this should support detectors, for now, just assume one
    return [dict(ident=0, anodes=d_anodes)]


class BoundingBox(object):
    def __init__(self):
        self.minp = None
        self.maxp = None

    @property
    def asdict(self):
        return dict(minp = self.minp, maxp = self.maxp)

    def __call__(self, p):
        if hasattr(p, "minp") and hasattr(p, "maxp"): # is a bb
            self(p.minp)
            self(p.maxp)
            return

        if self.minp is None:
            self.minp = dict(p)
            self.maxp = dict(p)
            return

        for c,v in self.minp.items():
            if p[c] < v:
                self.minp[c] = p[c]

        for c,v in self.maxp.items():
            if p[c] > v:
                self.maxp[c] = p[c]


    def center(self):
        return {c:0.5*(self.minp[c]+self.maxp[c]) for c in "xyz"}


class Ray(object):
    '''
    A ray is wrongly named but represents a line segment connecting two endpoints.
    '''
    def __init__(self, wire):
        self.head = numpy.asarray([wire['head'][c] for c in "xyz"])
        self.tail = numpy.asarray([wire['tail'][c] for c in "xyz"])
    @property
    def vector(self):
        return self.head - self.tail
    @property
    def center(self):
        return 0.5* (self.head + self.tail)


def pitch_summary(wires):
    '''
    Return pitch summary values [mean,rms,min,max,p0] 
    '''
    eks = numpy.asarray([1.0,0.0,0.0])
    zero = numpy.asarray([0.0, 0.0, 0.0])

    r0 = Ray(wires[0])
    pdir = numpy.cross(eks, r0.vector)
    pdir = pdir/numpy.linalg.norm(pdir)

    pmax=pmin=None

    prays = list()
    for w in wires:
        r = Ray(w)
        p = numpy.dot(pdir, r.center)
        prays.append((p,r))
    prays.sort()
    rays = [pr[1] for pr in prays]
    psum = psum2 = 0.0
    for count,(r1,r2) in enumerate(zip(rays[:-1], rays[1:])):
        p = numpy.dot(pdir,r2.center - r1.center)
        if count == 0:
            p0 = p
        if pmax is None:
            pmax = pmin = p
        else:
            pmax = max(pmax, p)
            pmin = min(pmin, p)
        psum += abs(p)
        psum2 += p*p
    n = len(wires)-1
    pmean = psum/n
    assert(pmean>0)
    pvar = (pmean*pmean - psum2/n)/(n-1)
    return pmean,  math.sqrt(abs(pvar)), pmin, pmax, p0

# old function
def pitch_mean_rms(wires):
    return pitch_summary(wires)[:-1]

def format_pitch(p):
    pmm = tuple([pp/units.mm for pp in p])
    return "(%.4f +/- %.6f [%.4f<%.4f], p0=%.4f) " % pmm

def wire_pitch_rays(wires):
    '''
    Return a rays for wire and pitch.

    '''
    wtot = numpy.sum(numpy.vstack([Ray(w).vector for w in wires]), axis=0)
    wmag = numpy.linalg.norm(wtot)
    wdir = wtot/wmag

    eks = numpy.asarray([1.0,0.0,0.0])
    pdir = numpy.cross(eks, wdir);
    pdir = pdir/numpy.linalg.norm(pdir)

    nwires = len(wires);
    ind1 = int(0.25*nwires);
    ind2 = int(0.75*nwires);
    
    w1 = Ray(wires[ind1]);
    w2 = Ray(wires[ind2]);
    c2c = w2.center - w1.center
    pmean = numpy.dot(pdir, c2c) / (ind2-ind1-1);

    w0 = Ray(wires[0]);       # tends to suffer from lack of precision
    c0 = w0.center
    c1 = c0 + pdir*pmean

    return dict(tail = {a:c0[i] for i,a in enumerate("xyz")},
                head = {a:c1[i] for i,a in enumerate("xyz")})


def summary_dict(store):

    '''
    Summarize the wires in store returning a dict

    Dict is a tree

    det -> anode -> face -> plane

    All three have key ["ident"] giving index number and "bb" giving
    bounding box with minp/maxp points

    Each plane has these keys:

        - pitch :: the wire pitch, also pmin, pmax and prms

        - nwires :: number of wires

    '''
    ret = list()
    for det in todict(store):
        ddict = dict()
        ddict['id'] = det['ident']
        ddict['anodes'] = list()
        dbb = BoundingBox()

        #print ('detector keys:',list(det.keys()))
        for anode in det['anodes']:
            adict = dict()
            adict["id"] = anode["ident"]
            adict['faces'] = list()
            #print ('\tanode %s keys: %s' % (anode['ident'], list(anode.keys())))

            abb = BoundingBox()

            faces = anode['faces']
            for face in faces:
                fdict = dict()
                fdict["id"] = face["ident"]
                fdict['planes'] = list()
                #print ('\t\tface %s keys: %s' % (face["ident"], list(face.keys())))

                fbb = BoundingBox()

                planes = face['planes']
                for plane in planes:
                    pdict = dict()
                    pdict['id'] = plane['ident']
                    #print ('\t\t\tplane %s keys: %s' % (plane["ident"], list(plane.keys())))

                    wires = plane['wires']
                    pdict['nwires'] = len(wires)
                    pdict['pitch'], pdict['prms'], pdict['pmin'], pdict['pmax'], pdict['p0'] = pitch_summary(wires)
                    pdict['prays'] = wire_pitch_rays(wires);
                    pbb = BoundingBox()
                    for wire in wires:
                        pbb(wire['head']);
                        pbb(wire['tail']);
                    fbb(pbb)
                    pdict['bb'] = pbb.asdict
                    fdict['planes'].append(pdict)

                abb(fbb)
                fdict['bb'] = fbb.asdict
                adict['faces'].append(fdict)
                
            dbb(abb)
            adict['bb'] = abb.asdict
            ddict['anodes'].append(adict)
            
        ddict['bb'] = dbb.asdict
        ret.append(ddict)
    return ret

def summary(store):
    '''
    Return a summary data structure about the wire store.
    '''
    lines = list()
    for det in todict(store):
        for anode in det['anodes']:
            for face in anode['faces']:
                bb = BoundingBox()
                pitches = list()
                for plane in face['planes']:
                    pitches.append(format_pitch(pitch_summary(plane['wires'])))
                    for wire in plane['wires']:
                        bb(wire['head']);
                        bb(wire['tail']);

                        # if anode['ident']==1 and face['ident']==1:
                        #     if wire['ident'] == 220:
                        #         print("--------special")
                        #         print(wire)
                        #         print("--------special")                            


                lines.append("anode:%d face:%d X=[%.2f,%.2f]mm Y=[%.2f,%.2f]mm Z=[%.2f,%.2f]mm" % \
                             (anode['ident'], face['ident'],
                              bb.minp['x']/units.mm, bb.maxp['x']/units.mm,
                              bb.minp['y']/units.mm, bb.maxp['y']/units.mm,
                              bb.minp['z']/units.mm, bb.maxp['z']/units.mm))
                for pind, plane in enumerate(face['planes']):
                    lines.append('\t%d: x=%.2fmm dx=%.4fmm n=%d pitch=%s' % \
                                 (plane['ident'],
                                  plane['wires'][0]['head']['x']/units.mm,
                                  (plane['wires'][0]['head']['x']-face['planes'][2]['wires'][0]['head']['x'])/units.mm,
                                  len(plane['wires']), pitches[pind]))
    return lines

def jsonnet_volumes(store,
                    danode=0.0, dresponse=10*units.cm, dcathode=1*units.m, volpat=None):
    '''
    Return a Jsonnet string suitable for copying to set
    params.det.volumes found in the pgrapher configuration.

    The "namepat" should be a string with format markers "{variable}"
    and will be formated with "detector", "anode" set to their
    numbers.

    The "d" arguments give the distance measured from the collection
    plane to each of these logical planes.
    '''

    voltexts = list()
    volpat = volpat or '''
        wires: {anode},
        name: anode{anode},
        faces: [ {faces} ],
'''
    facepat = "anode:{anodex}*wc.cm, cathode:{cathodex}*wc.cm, response:{responsex}*wc.cm"

    for idet, det in enumerate(todict(store)):
        for anode in det['anodes']:
            faces = anode['faces']
            assert (len(faces) <= 2)
            facetexts = ["null","null"]
            for face in faces:
                face_bbs = list()
                for plane in face['planes']:
                    bb = BoundingBox()
                    for wire in plane['wires']:
                        bb(wire['head']);
                        bb(wire['tail']);
                    face_bbs.append(bb)
                    continue    # plane
                uvw_x = [bb.minp['x'] for bb in face_bbs]
                sign = +1.0
                find = 0
                if uvw_x[0] < uvw_x[2]: # U below W means "back" face
                    sign = -1.0
                    find = 1
                
                xorigin = uvw_x[2]
                facetexts[find] = "\n            {" + facepat.format(
                    anodex = (xorigin + sign*danode)/units.cm,
                    responsex = (xorigin + sign*dresponse)/units.cm,
                    cathodex = (xorigin + sign*dcathode)/units.cm) + "}"
                continue        # face

            facetext = ','.join(facetexts)
            voltext = "    {" + volpat.format(anode=anode["ident"], faces=facetext) + "    }"
            voltexts.append(voltext)
            continue            # anode
        continue                # detector
    
    argstext   = "    local volumeargs = { danode:%f*wc.cm, dresponse:%f*wc.cm, dcathode:%f*wc.cm },\n" \
                                         % (danode/units.cm, dresponse/units.cm, dcathode/units.cm)
    volumetext = "    volumes: [\n" + ',\n'.join(voltexts) + "\n    ],\n";
    return argstext+volumetext;
    
