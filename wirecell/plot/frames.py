from wirecell import units
from wirecell.util import ario
from wirecell.util.plottools import lg10
import matplotlib.pyplot as plt
import numpy

def spectra(dat, out, tier='orig', interactive=False):
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
        
def wave(dat, out, tier='orig', interactive=False):
    '''
    Plot frames
    '''
    frames = sorted([f for f in dat.keys() if f.startswith(f'frame_{tier}')])

    for fname in frames:
        _,tag,num = fname.split("_")
        ticks = dat[f'tickinfo_{tag}_{num}']
        chans = dat[f'channels_{tag}_{num}']
        chmin = numpy.min(chans)
        chmax = numpy.max(chans)
        nchan = chmax-chmin+1;

        waves = dat[fname]      # (nch x ntick)
        waves = numpy.array((waves.T - numpy.median(waves, axis=1)).T, dtype='int16')
        chwaves = numpy.zeros((nchan, waves.shape[1]), dtype=int)
        for ind,ch in enumerate(chans):
            chwaves[ch-chmin] = waves[ind]
        maxtime = ticks[1]*waves.shape[1]

        print(chwaves.dtype, chwaves.shape)

        fig,ax = plt.subplots(1,1, figsize=(10,6))
        ax.set_title("Waveforms")
        im = ax.imshow(chwaves,
                       aspect='auto', interpolation='none',
                       extent=(0,maxtime/units.ms, chmax, chmin),
                       cmap='seismic', vmin=-25, vmax=25)

        ax.set_xlabel("time [ms]")
        ax.set_ylabel("channel")
        fig.colorbar(im, ax=ax)
        if interactive :
            plt.show()
        out.savefig(fig)
        
def wave_comp(datafile1, datafile2, out, tier='orig', channel=0, xrange=None, interactive=False):
    '''
    compare waveforms from files, assuming key names in two files are the same
    '''
    dat1 = ario.load(datafile1)
    dat2 = ario.load(datafile2)
    frames1 = sorted([f for f in dat1.keys() if f.startswith(f'frame_{tier}')])

    for fname in frames1:
        waves1 = dat1[fname]
        waves1 = numpy.array((waves1.T - numpy.median(waves1, axis=1)).T, dtype='int16')
        waves2 = dat2[fname]
        waves2 = numpy.array((waves2.T - numpy.median(waves2, axis=1)).T, dtype='int16')

        fig,ax = plt.subplots(1,1, figsize=(10,6))
        ax.set_title(f'Channel: {channel}')
        
        ax.plot(waves1[channel],'-',label=datafile1)
        ax.plot(waves2[channel],'o',label=datafile2)
        ax.set_xlabel("tick [0.5 $\mu$s]")
        ax.legend()
        if xrange is not None :
            ax.set_xlim(xrange)
        if interactive :
            plt.show()
        out.savefig(fig)