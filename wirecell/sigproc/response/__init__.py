#!/usr/bin/env python
'''
Functions related to responses.
'''
from wirecell import units

from . import schema

import math
import numpy
import collections


def electronics_no_gain_scale(time, gain, shaping=2.0*units.us):
    '''
    This version takes gain parameter already scaled such that the
    gain actually desired is obtained.  
    '''
    domain=(0, 10*units.us)
    if time <= domain[0] or time >= domain[1]:
        return 0.0

    time = time/units.us
    st = shaping/units.us
        
    from math import sin, cos, exp
    ret = 4.31054*exp(-2.94809*time/st) \
          -2.6202*exp(-2.82833*time/st)*cos(1.19361*time/st) \
          -2.6202*exp(-2.82833*time/st)*cos(1.19361*time/st)*cos(2.38722*time/st) \
          +0.464924*exp(-2.40318*time/st)*cos(2.5928*time/st) \
          +0.464924*exp(-2.40318*time/st)*cos(2.5928*time/st)*cos(5.18561*time/st) \
          +0.762456*exp(-2.82833*time/st)*sin(1.19361*time/st) \
          -0.762456*exp(-2.82833*time/st)*cos(2.38722*time/st)*sin(1.19361*time/st) \
          +0.762456*exp(-2.82833*time/st)*cos(1.19361*time/st)*sin(2.38722*time/st) \
          -2.6202*exp(-2.82833*time/st)*sin(1.19361*time/st)*sin(2.38722*time/st)  \
          -0.327684*exp(-2.40318*time/st)*sin(2.5928*time/st) +  \
          +0.327684*exp(-2.40318*time/st)*cos(5.18561*time/st)*sin(2.5928*time/st) \
          -0.327684*exp(-2.40318*time/st)*cos(2.5928*time/st)*sin(5.18561*time/st) \
          +0.464924*exp(-2.40318*time/st)*sin(2.5928*time/st)*sin(5.18561*time/st)
    ret *= gain
    return ret

def electronics(time, peak_gain=14*units.mV/units.fC, shaping=2.0*units.us):
    '''
    Electronics response function.

        - gain :: the peak gain value in [voltage]/[charge]

        - shaping :: the shaping time in Wire Cell system of units

        - domain :: outside this pair, the response is identically zero
    '''
    # see wirecell.sigproc.plots.electronics() for these magic numbers.
    if shaping <= 0.5*units.us:
        gain = peak_gain*10.146826
    elif shaping <= 1.0*units.us:
        gain = peak_gain*10.146828
    elif shaping <= 2.0*units.us:
        gain = peak_gain*10.122374
    else:
        gain = peak_gain*10.120179
    return electronics_no_gain_scale(time, gain, shaping)

electronics = numpy.vectorize(electronics)

def convolve(f1, f2):
    '''
    Return the simple convolution of the two arrays using FFT+mult+invFFT method.
    '''
    # fftconvolve adds an unwanted time shift
    #from scipy.signal import fftconvolve
    #return fftconvolve(field, elect, "same")
    s1 = numpy.fft.fft(f1)
    s2 = numpy.fft.fft(f2)
    sig = numpy.fft.ifft(s1*s2)

    return numpy.real(sig)

def _convolve(f1, f2):
    '''
    Return the simple convolution of the two arrays using FFT+mult+invFFT method.
    '''
    from scipy.signal import fftconvolve
    return fftconvolve(f1, f2, "same")



class ResponseFunction(object):
    '''
    A response function object holds the response wave function and metadata.

    Note: time is assumed to be in Wire Cell system of units (ns).  This is NOT seconds.
    '''
    def __init__(self, plane, region, pos, domainls, response, impact=None):
        plane = plane.lower()
        assert plane in 'uvw'
        self.plane = plane
        self.region = region
        self.pos = tuple(pos)
        self.domainls = domainls
        self.response = response
        self.times = numpy.linspace(*self.domainls)
        self.impact = impact


    def __call__(self, time):
        return numpy.interp(time, self.times, self.response)

    def resample(self, nbins):
        newls = (self.times[0], self.times[-1], nbins)
        newtimes = numpy.linspace(*newls)
        newresps = numpy.interp(newtimes, self.times, self.response)
        return self.dup(domainls=newls, response=newresps)

    def dup(self, **kwds):
        '''
        Return a new ResponseFunction which is a copy of this one and
        with any values in kwds overriding.
        '''
        return ResponseFunction(**dict(self.asdict, **kwds))

    @property
    def nbins(self):
        return self.domainls[2]

    @property
    def asdict(self):
        '''
        Object as a dictionary.
        '''
        return dict(plane=self.plane, region=self.region, pos=self.pos,
                    domainls=self.domainls, response=self.response.tolist(),
                    impact=self.impact)

    def shaped(self, gain=14*units.mV/units.fC, shaping=2.0*units.us, nbins=None):
        '''
        Convolve electronics shaping/peaking response, returning a new
        ResponseFunction.
        '''
        # use version defined above
        # from scipy.signal import fftconvolve
        if nbins is None:
            newfr = self.dup()
        else:
            newfr = self.resample(nbins)
        # integrate the current over the sample to get charge
        dt = newfr.times[1]-newfr.times[0]
        newfr.response = [r*dt for r in newfr.response]
        elecr = electronics(newfr.times, gain, shaping)
        newfr.response = convolve(elecr, newfr.response)
        return newfr

    def __str__(self):
        blah = "<ResponseFunction plane=%s region=%d domainls=%s pos=%s" % \
               (self.plane, self.region, self.domainls, self.pos)
        if self.impact is not None:
            blah += " impact=%f" % self.impact
        blah += ">"
        return blah


def group_by(rflist, field):
    '''
    Return a list of lists grouping by like values of the field.
    '''
    ret = list()
    for thing in sorted(set(getattr(d, field) for d in rflist)):
        bything = [d for d in rflist if getattr(d, field) == thing]
        ret.append(bything)
    return ret

def by_region(rflist, region=0):
    ret = [rf for rf in rflist if rf.region == region]
    ret.sort(key=lambda x: x.plane)
    return ret


def total_charge(rf):
    '''
    Integrate total charge in a current response.
    '''
    dt = (rf.times[1] - rf.times[0])
    if dt == 0.0:
        raise ValueError("Corrupt response function for plane %s, region %d" % (rf.plane, rf.region))
    itot = numpy.sum(rf.response)
    return dt*itot

def normalize(rflist, plane='w', region=0, impact=None):
    '''
    Return new rflist with all responses normalized to be in Ampere
    and assuming a single electron was drifting.

    The collection signal on the given plane and region is used to
    normalize.  If impact is an impact distance, or a list of them
    then the average collection signal is used.  If not given, all
    impacts for the given plane/region are used.
    '''
    toaverage = [rf for rf in rflist if rf.plane == plane and rf.region == region]
    if impact is not None:
        if not isinstance(impact, collections.Sequence):
            impact = [impact]
        toaverage = [rf for rf in toaverage if rf.impact in impact]

    num = len(toaverage)
    if 0 == num:
        msg = "No fields to average out of %d for nomalize(%s, %d, %s)" % (len(rflist), plane, region, impact)
        raise ValueError(msg)

    qtot = sum([total_charge(rf) for rf in toaverage])
    qavg = qtot/num
    scale = -units.eplus/qavg

    out = list()
    for rf in rflist:
        newrf = rf.dup(response = rf.response*scale)
        out.append(newrf)
    return out


def _average(fine):
    '''
    Average fine-grained response functions over multiple impact
    positions in the same plane and wire region.

    Return list of new response.ResponseFunction objects ordered by
    plane, region.
    '''
    ret = list()
    for inplane in group_by(fine, 'plane'):
        byregion = group_by(inplane, 'region')
        noigeryb = list(byregion)
        noigeryb.reverse()

        regions = [rflist[0].region for rflist in byregion]
        center = max(regions)

        # warning: this makes detailed assumptions about where the impact points are!
        for regp,regm in zip(byregion[center:], noigeryb[center:]):
            regp.sort(key=lambda x: x.impact)
            regm.sort(key=lambda x: x.impact)

            tot = numpy.zeros_like(regp[0].response)
            count = 0
            for one in regp + regm:       # sums 2 regions!
                tot += one.response
                count += 1
            tot *= 2.0                                 # reflect across wire region
            count *= 2
            tot -= regp[0].response + regm[0].response # share region boundary path
            count -= 2
            tot -= regp[-1].response + regm[-1].response # don't double count impact=0
            count -= 2
            tot /= count
            dat = regp[0].dup(response=tot, impact=None)
            ret.append(dat)
        continue
    return ret

def average(fine):
    '''
    Average fine-grained response functions over multiple impact
    positions in the same plane and wire region.  It assumes an odd
    number of regions and a half-populated impact positions per region
    such that the first impact is exactly on a wire and the impact is
    exactly on a half-way line between neighboring wires.

    Return list of new response.ResponseFunction objects ordered by
    plane, region which cover the same regions.
    '''
    coarse = list()
    for inplane in group_by(fine, 'plane'):
        byregion = group_by(inplane, 'region')

        # for each region, we need to take impacts from the region on the other
        # side of center.  So march down the reverse while we march up the
        # original.
        noigeryb = list(byregion)
        noigeryb.reverse()

        for regp, regm in zip(byregion, noigeryb):
            # Assure each region is sorted so first is impact=0
            regp.sort(key=lambda x: abs(x.impact))
            regm.sort(key=lambda x: abs(x.impact))

            tot = numpy.zeros_like(regp[0].response)

            nimpacts = len(regp)
            binsize = [1.0]*nimpacts      # unit impact bin size
            binsize[0] = 0.5;             # each center impact only covers 1/2 impact bin
            binsize[-1] = 0.5;            # same for the edge impacts 

            for impact in range(nimpacts):
                rp = regp[impact]
                rm = regm[impact]
                tot += binsize[impact]*(rp.response + rm.response)

            # normalize by total number of impact bins
            tot /= 2*(nimpacts-1)
            dat = regp[0].dup(response=tot, impact=None)
            coarse.append(dat)
        continue
    return coarse




def field_response_spectra(rflist):
    '''
    Return a tuple of response spectra as collection of per-plane
    matrices in channel periodicity vs frequency.  The rflist is both
    averaged over impacts (if needed) and normalized.
    '''
    impacts = set([rf.impact for rf in rflist])
    if len(impacts) > 1:
        rflist = average(rflist)
    rflist = normalize(rflist)

    ret = list()

    byplane = group_by(rflist, 'plane')
    for inplane in byplane:
        inplane.sort(key=lambda x: x.region)
        responses = [rf.response for rf in inplane]
        rows = list(responses)
        rows.reverse()          # mirror 
        rows += responses[1:]   # don't double add region==0
        mat = numpy.asarray(rows)
        spect = numpy.fft.fft2(mat, axes=(0,1))
        ret.append(spect)
    return tuple(ret)
    
def plane_impact_blocks(rflist, eresp = None):
    '''
    Return a field responses as a number of matrices blocked by impacts.

    Returns a triple of arrays of shape (Nimpacts, Nregion, Ntbins).

    If eresp is given, it is convolved with each response function.
    '''

    # symmetry: Response on wire 0 due to path at wire region i,
    # impact j is same as Response on wire 0 due to path at wire
    # region -i, impact -j.

    # symmetry: Response on Wire 0 due to path at wire region i,
    # impact j is same as response on wire i, due to path at wire
    # region 0, impact j.

    ret = list()
    byplane = group_by(rflist, 'plane')
    for inplane in byplane:
        byimpact = group_by(inplane, 'impact')
        impacts = [d[0].impact for d in byimpact]
        nimpacts = len(impacts)
        regions = list(set([rf.region for rf in inplane]))
        regions.sort()
        nregions = len(regions)
        ntbins = len(inplane[0].response)
        pib_shape = (nimpacts, nregions, ntbins)

        pib = numpy.zeros(pib_shape)
        for inimpact in byimpact:
            impact_index = impacts.index(inimpact[0].impact)
            for inregion in inimpact:
                region_index = regions.index(inregion.region)
                pib[impact_index, region_index] = inregion.response
        ret.append(pib)
    return ret


# pibs
class PlaneImpactBlocks(object):
    '''
    Organize responses into per (plane,impact) and make available as
    array blocks of shape (Nregion,Ntbins).

    There are two symmetries for response Resp(w,r,i) on wire w for
    path near wire region r on impact i.

        - Due to reciprocity: Resp(w=0,r=N,i) = R(w=N,r=0,i)

        - Due to geometry: R(w=0,r=N,i=M) = R(w=0,r=-N,i=-M)

    See the functions `plots.plane_impact_blocks_full()` and
    `plots.plane_impact_blocks()` for visualizing this data.  In
    particular check for continuous patterns of responses across
    different impact positions in the first and check that U-plane,
    high-positive impact position puts most response on wires 0 and 1
    and high-negative impact positions puts most response on wires 0
    and -1.
    '''
    def __init__(self, rflist, xstart = 0.0*units.cm):

        onerf = rflist[0]
        self.ntbins = len(onerf.response)
        self.tmin = onerf.times[0]
        self.tbin = onerf.times[1]-onerf.times[0]
        self.trange = self.tbin*self.ntbins
        self.tmax = self.trange + self.tmin

        self.xstart = xstart # x position at start of field response drift


        self.plane_keys = sorted(set([rf.plane for rf in rflist]))
        self.region_keys = sorted(set([rf.region for rf in rflist]))
        self.impact_keys = sorted(set([rf.impact for rf in rflist] + [-rf.impact for rf in rflist]))

        # Organize flat rflist into tree [plane][impact][region]
        tree = dict()
        byplane = group_by(rflist, 'plane') # uvw
        for inplane in byplane:
            plane_letter = inplane[0].plane
            tree[plane_letter] = tree_plane = dict()
            byimpact = group_by(inplane, 'impact')
            for inimpact in byimpact:
                # WARNING: Garfield seems to measure either wire
                # region number xor impact position in a different
                # direction than is assumed here.  Garfield impact
                # positions are always positive.
                impact = -1*inimpact[0].impact
                assert impact <= 0.0

                tree_plane[-impact] = tree_impact_pos = dict()
                tree_plane[impact] = tree_impact_neg = dict()
                
                byregion = group_by(inimpact, 'region')
                for inregion in byregion:
                    assert len(inregion) == 1
                    rf = inregion[0]
                    region = rf.region
                    tree_impact_pos[region] = rf
                    tree_impact_neg[-region] = rf
        self._tree = tree
        self._by_plane_impact = dict()

    def region_block(self, plane, impact):
        '''
        Return an array shaped (Nregions, Ntbins) for the given plane
        and impact.  Row=0 corresponds to the highest region (wire 10).
        '''
        key = (plane,impact)    # cache the built array
        try:
            return self._by_plane_impact[key]
        except KeyError:
            pass
        ppi = self._tree[plane][impact]
        rfs = [ppi[r] for r in self.region_keys]
        mat = numpy.zeros((len(self.region_keys), len(rfs[0].response)))
        for row,rf in enumerate(rfs):
            mat[row] = rf.response
        self._by_plane_impact[key] = mat
        return mat
            
    def response(self, plane, impact, region):
        return self._tree[plane][impact][region].response

class foo():
        
    def __init__(self):


        self.region_center = self.nregions // 2
        
    def __call__(self, plane, impact, region=None):
        '''
        Return a response block by plane (0,1,2) and by impact
        (-4,...,0,...,5) and optionally by region (-10,...,0,...,10).

        If region is None, return corresponding (Nregions,Ntbins)
        array.  If region is given, return (Ntbins) array.  The impact
        (and region) numbers may be negative.
        '''
        pib = self.pibs[plane]
        if region is None:      # 2D
            if impact >= 0:
                return pib[impact]
            return numpy.flipud(pib[-impact])
        # 1D
        impact,region = self.impact_region_numbers_to_indices(impact, region)
        return pib[impact, region]

    def impact_region_numbers_to_indices(self, impact, region):
        if impact < 0:          # must reflect
            impact *= -1
            region *= -1
        region + self.region_center
        return (impact,region)
        

    @property
    def impact_range(self):
        max_impact = len(self.pibs[0]) - 1
        return range(-max_impact, max_impact) # skip highest as it's shared with lowest of lower region
    @property
    def region_range(self):
        nhalf_regions = self.nregions//2
        return range(-nhalf_regions, nhalf_regions+1) # inclusive

    @property
    def nregions(self):
        return self.pibs[0][0].shape[0]
    @property
    def ntbins(self):
        return self.pibs[0][0].shape[1]
        


def response_spect_nominal(rflist, gain, shaping, tick=0.5*units.us):
    '''
    Return the a response matrix such as passed to `deconvolve()`.

    Only the frequencies corresponding to a sampling period of `tick`
    are retained.  
    '''
    first = rflist[0]
    frm = field_response_spectra(rflist)

    elect = electronics(first.times, gain, shaping)
    elesp = numpy.fft.fft(elect)

    # number of frequencies to keep to correspond to downsampled tick
    Nhave = first.nbins
    Nkeep = int(round(first.nbins/(2.0*tick/(first.times[1]-first.times[0]))))

    # chop out the "middle" frequencies around the nominal Nyquist freq.
    frm_chopped = [numpy.delete(f, range(Nkeep, Nhave-Nkeep), axis=1) for f in frm]
    ele_chopped = numpy.delete(elesp, range(Nkeep, Nhave-Nkeep))

    resp = [ele_chopped*f for f in frm_chopped]
    return resp
        

def filter_expower(sig, power, nbins, nyquist):
    '''
    Return a Fourier space filter function:

    filter = exp(-(freq/sig)^(power))

    The filter function is returned as an array `nbins` long covering
    the low half of the Fourier domain up to the given `nyquist`
    "frequency".

    The `sig` and `nyquist` parameters are in units of the relevant
    "frequency" for the given Fourier domain.  For time-domain Fourier
    this is likely Hz.  For channel-domain this is likely in units of
    per pitch (unitless).  Caller assures this consistency.
    '''
    freqs = numpy.linspace(0, nyquist, nbins)
    def filt(f):
        return math.exp(-0.5*((f/sig)**power))
    filt = numpy.vectorize(filt)
    return filt(freqs)

    
def filters(nticks=9600, tick=0.5*units.us, npitches=3000, pitch=1.0):
    '''
    Return (fu,fv,fw,fc) filters.

    See `filter_expower()` for details.
    '''
    tick_seconds = tick/units.s
    nyquist_hz = 1.0/(2*tick_seconds)

    # note, these parameters are scaled from what is in the prototype
    # to be implicitly in Hz instead of MHz.
    #
    # These parameters were found by applying a Weiner-inspired filter
    # to a toy simulation using 2D microboone response and electronics
    # functions and noise model.  They may need to be re-evaluated for
    # other detectors.
    fu = filter_expower(2*1.43555e+07/200.0, 4.95096e+00, nticks, nyquist_hz)
    fv = filter_expower(2*1.47404e+07/200.0, 4.97667e+00, nticks, nyquist_hz)
    fw = filter_expower(2*1.45874e+07/200.0, 5.02219e+00, nticks, nyquist_hz)

    nyquist_pp = 1.0/(2*pitch)  # pp = per pitch

    # note, this parameter is scaled to move the somewhat bogus 0.5
    # (us) number that was used in the prototype out of the "freq" and
    # into the "sig".  In microboone, this filter appears to be only
    # needed for simulation.
    fc = filter_expower((1.4*0.5)/math.sqrt(math.pi), 2.0, npitches, nyquist_pp)

    return (fu, fv, fw, fc)


def deconvolve(Mct, Rpf, Ff, Fp):
    '''
    Return a matrix like Mct which is deconvolved by Rpf and filtered
    with the Ff and Fp.

    Indices are c=channel, t=time, p=periodicity, f=frequency.

    Mct is the measured ADC in a plane as a matrix of Nchannel rows
    and Nticks columns.

    Rpf is the wire periodicity and frequency space 2D Fourier
    transform of the field response * electronics response.  It is
    assumed to contain only the four corners up the time and wire
    Nyquist frequencies of Mct.

    Ff is a frequency space filter.

    Fp is the channel periodicity space filter.
    '''

    nchan,ntick = Mct.shape
    nperi,nfreq = Rpf.shape

    Mpf = numpy.fft.fft2(Mct, axes=(0,1))
    Mpf = numpy.delete(Mpf, range(nperi/2, nchan-nperi/2), axis=0)
    Mpf = numpy.delete(Mpf, range(nfreq/2, ntick-nfreq/2), axis=1)

    Spf = Mpf/Rpf * Fp[:nperi].reshape(1,nperi).T
    Scf = numpy.fft.ifft2(Spf, axes=(1,))
    Scf = Scf * Ff[:nfreq]
    Sct = numpy.fft.ifft2(Scf, axes=(0,))
    return Sct


def schematorf1d(fr):
    '''
    Convert response.schema objects to 1D ResponseFunction objects.

    Fixme: this has not yet been validated.
    '''
    ret = list()
    for pr in fr.planes:
        for path in pr.paths:
            region = int(round(path.pitchpos/pr.pitch))
            pos = (pr.wirepos, pr.pitchpos)
            nsamples = len(path.current)
            times = (fr.tstart, nsamples*fr.period, nsamples)
            impact = pathc.pitchpos - region*pr.pitch
            rf = ResponseFunction(pr.planeid, region, pos, times, path.current, impact)
            ret.append(rf)
    return ret
                                      


def rf1dtoschema(rflist, origin=10*units.cm, speed = 1.114*units.mm/units.us):
    '''
    Convert the list of 1D ResponseFunction objects into
    response.schema objects.

    The "1D" refers to the drift paths starting on a line in 2D space.

    Because it is 1D, all the pitch and wire directions are the same.
    '''
    #rflist = normalize(rflist)

    anti_drift_axis = (1.0, 0.0, 0.0)
    one = rflist[0]             # get sample times
    period = (one.times[1] - one.times[0])
    tstart = one.times[0]

    planes = list()
    byplane = group_by(rflist, 'plane')
    for inplane in byplane:
        letter = inplane[0].plane
        planeid = "uvw".index(letter)
        onetwo = [rf for rf in inplane if rf.impact == 0.0 and (rf.region == 0 or rf.region==1)]
        pitch = abs(onetwo[0].pos[0] - onetwo[1].pos[0])
        location = inplane[0].pos[1]
        inplane.sort(key=lambda x: x.region*10000+x.impact)

        paths = list()
        for rf in inplane:
            pitchpos = (rf.region*pitch + rf.impact)
            wirepos = 0.0
            par = schema.PathResponse(rf.response, pitchpos, wirepos)
            paths.append(par)

        plr = schema.PlaneResponse(paths, planeid, location, pitch)
        planes.append(plr)
    return schema.FieldResponse(planes, anti_drift_axis, origin, tstart, period, speed)





def write(rflist, outputfile = "wire-cell-garfield-response.json.bz2"):
    '''
    Write a list of response functions to file.
    '''
    import json
    text = json.dumps([rf.asdict for rf in rflist])
    if outputfile.endswith(".json"):
        open(outputfile,'w').write(text)
        return
    if outputfile.endswith(".json.bz2"):
        import bz2
        bz2.BZ2File(outputfile, 'w').write(text)
        return
    if outputfile.endswith(".json.gz"):
        import gzip
        gzip.open(outputfile, "wb").write(text)
        return
    raise ValueError("unknown file format: %s" % outputfile)
# fixme: implement read()

def line(rflist, normalization=13700*units.eplus):
    '''
    Assuming an infinite track of `normalization` ionization electrons
    per pitch which runs along the starting points of the response
    function paths, calculate the average response on the central wire
    of each plane.  The returned responses will be normalized such
    that the collection response integrates to the given normalization
    value if nonzero.
    '''

    impacts = set([rf.impact for rf in rflist])
    if len(impacts) > 1:
        rflist = average(rflist)
    byplane = group_by(rflist, 'plane')
    
    # sum across all impact positions assuming a single point source is
    # equivalent to summing across a perpendicular line source for a single,
    # central wire.
    ret = list()
    for inplane in byplane:
        first = inplane[0]
        tot = numpy.zeros_like(first.response)
        for rf in inplane:
            tot += rf.response
        dat = first.dup(response=tot, impact=None, region=None)
        ret.append(dat)

    # normalize to user's amount of charge
    if normalization > 0.0:
        for rf in ret:
            rf.response *= normalization

    return ret

