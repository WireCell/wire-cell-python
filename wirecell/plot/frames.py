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
    frames = sorted([f for f in dat.keys() if f.startswith(f'frame_{tier}')])

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

def comp1d(datafile1, datafile2, out, name='wave', tier='orig', chmin=0, chmax=1, unit='ADC', xrange=None, interactive=False):
    '''
    compare waveforms from files, assuming key names in two files are the same
    '''
    if name not in ['wave', 'spec']:
        raise('name not in [\'wave\', \'spec\']!')

    if unit == 'ADC':
        uscale = 1
        dtype = 'int16'
    else:
        uscale = getattr(units, unit)
        dtype = float

    dat1 = ario.load(datafile1)
    dat2 = ario.load(datafile2)
    frames1 = sorted([f for f in dat1.keys() if f.startswith(f'frame_{tier}')])

    def extract(dat, chmin, chmax):
        frame = dat[fname]
        print(frame.shape)
        frame = frame[chmin:chmax]
        frame = numpy.array((frame.T - numpy.median(frame, axis=1)).T, dtype=dtype)
        if dtype == float:
            frame /= 1.0*uscale
        if name == 'spec':
            frame = numpy.abs(numpy.fft.fft(frame, norm=None))
        return numpy.mean(frame, axis=0)

    for fname in frames1:
        _,tag,num = fname.split("_")
        chans = dat1[f'channels_{tag}_{num}']
        offset = numpy.min(chans)
        print(f'chan offset: {offset}')
        waves1 = extract(dat1, chmin-offset, chmax-offset)
        waves2 = extract(dat2, chmin-offset, chmax-offset)

        fig,ax = plt.subplots(1,1, figsize=(10,6))
        ax.set_title(f'Channel: {chmin} - {chmax}')
        
        ax.plot(waves1,'-',label=datafile1+f' std:{numpy.std(waves1):.2f}')
        ax.plot(waves2,'o',label=datafile2+f' std:{numpy.std(waves2):.2f}')
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
    
    # layout = "constrained"
    layout = "tight"            # this gives warning but closer to what I want.
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

    fig.savefig(output, format=format, bbox_inches='tight')
    
