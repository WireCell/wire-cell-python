#!/usr/bin/env python
import os
from wirecell import units
from wirecell.util import ario
from wirecell.util.plottools import lg10
import matplotlib.pyplot as plt
import numpy

def spectra(dat, out, tier='orig', unit='ADC', range=25, interactive=False):
    '''
    Plot per-channel spectra of fp['frame_{tier}*'] to out
    '''
    frames = sorted([f for f in dat.keys() if f.startswith(f'frame_{tier}')])

    for fname in frames:
        _,tag,num = fname.split("_")
        ticks = dat[f'tickinfo_{tag}_{num}']
        chans = dat[f'channels_{tag}_{num}']
        chmin = numpy.min(chans)
        chmax = numpy.max(chans)
        nchan = chmax-chmin+1;

        Fmax_MHz = (1/ticks[1]) / units.MHz
        waves = dat[fname]      # (nch x ntick)
        specs = numpy.fft.fft(waves, axis=1)

        hsize = specs.shape[1]//2
        hspecs = numpy.abs(specs[:, :hsize])
        #hspecs = lg10(hspecs)
        hspecs[:,0] = hspecs[:,1] # cheat to ignore DC
        tot = numpy.zeros_like(hspecs[0]);
        chspecs = numpy.zeros((nchan, hsize))
        for ind,ch in enumerate(chans):
            chspecs[ch-chmin] = hspecs[ind]
            tot += hspecs[ind]
        avg = tot / len(chans)

        fig,ax = plt.subplots(1,1)
        ax.set_title("Per channel spectra")
        im = ax.imshow(chspecs, aspect='auto', interpolation='none',
                       extent=(0,Fmax_MHz/2, chmax, chmin))

        ax.set_xlabel("frequency [MHz]")
        ax.set_ylabel("channel")
        fig.colorbar(im, ax=ax)
        out.savefig(fig)

        fig,ax = plt.subplots(1,1)
        ax.set_title("Average spectra")
        freqs = numpy.linspace(0, Fmax_MHz/2, avg.size, endpoint=False)
        ax.plot(freqs, avg)
        ax.set_xlabel("frequency [MHz]")
        if interactive :
            plt.show()
        out.savefig(fig)
        
def wave(dat, out, tier='orig', unit='ADC', vmm=25, interactive=False):
    '''
    Plot frames
    '''
    frame_keys = [f for f in dat.keys() if f.startswith('frame_')]
    frames = sorted([f for f in frame_keys if f.startswith(f'frame_{tier}')])
    if not frames:
        found = ', '.join(frame_keys)
        msg = f'No frames of tier "{tier}": found: {found}'
        raise IOError(msg)

    if unit == 'ADC':
        uscale = 1
        dtype = 'int16'
    else:
        uscale = getattr(units, unit)
        dtype = float



    for fname in frames:
        _,tag,num = fname.split("_")
        print(f'frame "{tag}" #{num}')
        ticks = dat[f'tickinfo_{tag}_{num}']
        chans = dat[f'channels_{tag}_{num}']
        chmin = numpy.min(chans)
        chmax = numpy.max(chans)
        nchan = chmax-chmin+1;

        waves = dat[fname]      # (nch x ntick)
        waves = numpy.array((waves.T - numpy.median(waves, axis=1)).T, dtype=dtype)
        if dtype == float:
            waves /= 1.0*uscale
        chwaves = numpy.zeros((nchan, waves.shape[1]), dtype=dtype)
        for ind,ch in enumerate(chans):
            chwaves[ch-chmin] = waves[ind]
        maxtime = ticks[1]*waves.shape[1]

        fig,(ax,ax2) = plt.subplots(1,2, figsize=(10,6), sharey=True,
                                    gridspec_kw={'width_ratios': [5, 1]})
        ax.set_title("Waveforms")
        im = ax.imshow(chwaves,
                       aspect='auto', interpolation='none',
                       extent=(0,maxtime/units.ms, chmax, chmin),
                       cmap='seismic', vmin=-vmm, vmax=vmm)

        ax.set_xlabel("time [ms]")
        ax.set_ylabel("channel")
        fig.colorbar(im, ax=ax, label=unit)

        chwaves = numpy.array(chwaves, dtype=float)
        chss = numpy.sum(chwaves ** 2, axis=1)
        print(numpy.any(chss < 0))
        rms = numpy.sqrt(chss/chwaves.shape[1])
        chans = numpy.linspace(chmin, chmax+1, chwaves.shape[0], endpoint=True)
        ax2.step(rms, chans)
        ax2.set_xlabel(f'[{unit}]')
        ax2.set_title("RMS")
        if interactive :
            plt.show()
        out.savefig(fig)


def comp1d(datafiles, out, name='wave', frames='orig',
           chbeg=0, chend=1, unit='ADC', xrange=None,
           interactive=False, transforms=(),
           markers = ['o', '.', ',', '+', 'X', "*"]
           ):
    '''Compare similar waveforms across datafiles.

    Comparisons are made based on common frame array names in the files.

    The "datafiles" is a list of file name or ario-like file objects.

    The "frames" argument specifies the frames either as an explicit
    list of frame array names or by a "tier" to match as in the
    pattern 'frame_<tier>_<ident>'.

    The "out" argument is a PdfPages like object.

    The "name" gives the type of comparison plot to produce.

    The range of channel idents will be in the half-open range ["chbeg","chend").

    The "baseline" sets if and how a re-baselining is performed.

    '''

    # Head-off bad calls
    if name not in ['wave', 'spec']:
        raise('name not in [\'wave\', \'spec\']!')

    # when run from a historical test it is common to have long,
    # common path prefixes in the input files.  Below we apply trunc()
    # to shorten the path names in the legend.
    pre = os.path.commonpath(datafiles)
    def trunc(path):
        if path.startswith(pre):
            return path[len(pre)+1:]
        return path

    # Open data files if we have a file name, else assume an ario-like
    # file object.
    dats = list()
    for dat in datafiles:
        if isinstance(dat, str):
            dat = ario.load(dat)
        dats.append(dat)

    # Apply tier selector if frames is a string
    if isinstance(frames, str):
        fnames = set()
        for dat in dats:
            fnames.update([n for n in dat.keys() if n.startswith(f'frame_{frames}')])
        fnames = list(fnames)
        fnames.sort()
        frames = fnames
        
    # Treat ADC special
    if unit == 'ADC':
        uscale = 1
        dtype = 'int16'
    else:
        uscale = getattr(units, unit)
        dtype = float

    # Note, channel numbers are in general opaquely defined.  We must
    # not assume anything about a channel array's order, monotonicity,
    # density, etc.  Note, each dat may have a different channel set.
    def channel_selection(fname, dat):
        'Return True/False for each frame array row if it is a selected channel'
        _,tag,num = fname.split("_")
        chans = dat[f'channels_{tag}_{num}']
        return numpy.logical_and(chans >= chbeg, chans < chend)

    # Extract the thing to plot.
    def extract(fname, dat):
        frame = numpy.array(dat[fname], dtype=dtype)
        chans = channel_selection(fname, dat)
        frame = frame[chans,:]
        # frame = numpy.array((frame.T - numpy.median(frame, axis=1)).T, dtype=dtype)
        if "median" in transforms:
            fmed = numpy.median(frame, axis=1)
            frame = (frame.T - fmed).T
        if "mean" in transforms:
            fmu = numpy.mean(frame, axis=1)
            frame = (frame.T - fmu).T
        if dtype == float:
            frame /= 1.0*uscale

        if "ac" in transforms:  # treat special so we ac-couple either spec or wave
            cspec = numpy.fft.fft(frame)
            cspec[:,0] = 0      # set all zero freq bins to zero
            if name == "spec":
                return numpy.mean(numpy.abs(cspec), axis=0)
            wave = numpy.fft.ifft(cspec)
            return numpy.mean(numpy.real(wave), axis=0)

        if name == 'spec':
            frame = numpy.abs(numpy.fft.fft(frame))
            return numpy.mean(frame, axis=0)
        return numpy.mean(frame, axis=0)

    for fname in frames:

        fig,ax = plt.subplots(1,1, figsize=(10,6))
        tit = f'{chbeg} <= channel < {chend}'
        if transforms:
            tit += ' (' + ', '.join(transforms) + ')'
        ax.set_title(tit)

        for ind, dat in enumerate(dats):
            thing = extract(fname, dat)
            marker = markers[ind%len(markers)]
            
            tit = trunc(dat.path)+f'\nN:{thing.size} mean:{numpy.mean(thing):.2f} std:{numpy.std(thing):.3f}'
            ax.plot(thing, marker, label=tit)

        ax.set_xlabel("tick [0.5 $\mu$s]")
        ax.legend()
        if xrange is not None :
            ax.set_xlim(xrange)
        if interactive :
            plt.show()
        out.savefig(fig)

def channel_correlation(datafile, out, tier='orig', chmin=0, chmax=1, unit='ADC', interactive=False):
    '''
    check channel correlations
    '''
    if unit == 'ADC':
        uscale = 1
        dtype = 'int16'
    else:
        uscale = getattr(units, unit)
        dtype = float

    dat = ario.load(datafile)
    frames = sorted([f for f in dat.keys() if f.startswith(f'frame_{tier}')])

    def extract(chmin, chmax):
        frame = dat[fname]
        # print(frame.shape)
        frame = frame[chmin:chmax]
        frame = numpy.array((frame.T - numpy.median(frame, axis=1)).T, dtype=dtype)
        if dtype == float:
            frame /= 1.0*uscale
        return frame

    for fname in frames:
        _,tag,num = fname.split("_")
        chans = dat[f'channels_{tag}_{num}']
        offset = numpy.min(chans)
        print(f'chan offset: {offset}')
        chmin = chmin - offset
        chmax = chmax - offset
        frame = extract(chmin, chmax)

        fig,ax = plt.subplots(1,1, figsize=(10,6))
        ax.set_title(f'{datafile}')
        
        ax.imshow(numpy.corrcoef(numpy.abs(frame)), cmap=plt.get_cmap("bwr"), interpolation = 'none', clim=(-1,1))
        ax.set_xlabel("Channel")
        if interactive :
            plt.show()
        out.savefig(fig)


def frame_means(array, channels, cmap, format, output, aname, fname):
    '''
    Plot frames and their channel-wise and tick-wise means
    '''

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    
    # this gives more wasted white space but less likely to have axes labels overlapping
    layout = "constrained" 
    # this gives warning but closer to what I want.
    # layout = "tight"
    fig = plt.figure(layout=layout, figsize=(10,8))
    fig.suptitle(f'array "{aname}" from {fname}\nand time/channel projected means')
    # base + 
    # [0:mean, 1:image, 2:colorbar]
    # [      , 4:mean,            ]
    # x3

    nplns=3
    nrows=2
    ncols=2
    gridspec = GridSpec(nplns*nrows+1, ncols,
                        figure=fig,
                        height_ratios=[1,3,1,3,1,3,1], width_ratios=[1,30],
                        left=0.05, right=0.95, hspace=0.0001, wspace=0.0001)
    def gs(pln, row, col):
        return gridspec[(pln*nrows+1)*ncols + row*ncols + col]

    steerx = None
    steerc = None
    steert = None

    normalizer=Normalize(numpy.min(array), numpy.max(array))
    cb = cm.ScalarMappable(norm=normalizer)
    
    aximgs = list()
    for pln, letter in enumerate("UVW"):

        pgs = lambda r,c: gs(pln, r, c)

        base = pln*6

        if steerx is None:
            aximg = fig.add_subplot(pgs(0, 1))
            steerx = aximg
        else:
            aximg = fig.add_subplot(pgs(0, 1), sharex=steerx)

        aximg.set_axis_off()
        aximgs.append(aximg)

        if steerc is None:
            axmu0 = fig.add_subplot(pgs(0, 0), sharey=aximg)
            steerc = axmu0
        else:
            axmu0 = fig.add_subplot(pgs(0, 0), sharey=aximg, sharex=steerc)

        if steert is None:
            axmu1 = fig.add_subplot(pgs(1, 1), sharex=steerx)
            steert = axmu1
        else:
            axmu1 = fig.add_subplot(pgs(1, 1), sharex=steerx, sharey=steert)
        axmus = [axmu1, axmu0]

        if pln == 0:
            plt.setp( axmu1.get_xticklabels(), visible=False)
            axmu0.xaxis.tick_top()
            axmu0.tick_params(axis="x", labelrotation=90)
        if pln == 1:
            plt.setp( axmu1.get_xticklabels(), visible=False)
            axmu0.set_ylabel('channels')
            plt.setp( axmu0.get_xticklabels(), visible=False)
        if pln == 2:
            axmu1.set_xlabel("ticks")
            plt.setp( axmu0.get_xticklabels(), visible=False)

        axmu0.ticklabel_format(useOffset=False)
        axmu1.ticklabel_format(useOffset=False)

        crange = channels[pln]
        achans = numpy.array(range(*crange))
        aticks = numpy.array(range(array.shape[1]))
        xses = [aticks, achans]
        plane = array[achans,:]
        im = aximg.imshow(plane, cmap=cmap, norm=normalizer,
                          extent=(aticks[0], aticks[-1], crange[1], crange[0]),
                          interpolation="none", aspect="auto")

        for axis in [0,1]:
            mu = plane.sum(axis=axis)/plane.shape[axis]
            axmu = axmus[axis]
            xs = xses[axis]

            if axis: 
                axmu.plot(mu, xs)
            else:
                axmu.plot(xs, mu)
                      
            
    axcb = fig.add_subplot(gridspec[1])
    fig.colorbar(cb, cax=axcb, ax=aximgs, cmap=cmap, location='top')

    fig.savefig(output, format=format)
    
