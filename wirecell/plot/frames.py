from wirecell import units
from wirecell.util.plottools import lg10
import matplotlib.pyplot as plt
import numpy

def spectra(dat, out, tier='orig'):
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
        out.savefig(fig)
        
def wave(dat, out, tier='orig'):
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
        out.savefig(fig)

        
