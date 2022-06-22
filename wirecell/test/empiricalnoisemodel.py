from wirecell import units
import matplotlib.pyplot as plt
import numpy

def plot(dat, out):
    '''
    Make plots into PdfPages-like object "out" from data loaded from
    the file produced by test_empiricalnoisemodel.
    '''
    def keyswith(pre):
        return sorted([k for k in dat.keys() if k.startswith(pre)])


    planes = dat['planes']
    nsamples = dat['nsamples']
    wirelen = dat['wirelens']
    periods = dat['periods']

    fig,ax = plt.subplots(1,1)
    for ind, freqname in enumerate(keyswith("freqs_")):
        num = freqname.split("_")[1]
        freq = dat[freqname]
        amps = dat[f'amps_{num}']
        E = numpy.sum(amps**2)
        l = wirelen[ind]/units.m
        p = periods[ind]/units.us
        pln = "uvw"[planes[ind]]
        ax.plot(freq / units.MHz, amps,
                label=f'{pln}: p={p}us w={l}m ns={nsamples[ind]} nf={freq.size} E={E:.2e}')
    ax.set_title("Input spectra")
    ax.set_xlabel("frequency [MHz]")
    ax.set_ylabel("amp")
    plt.legend()
    out.savefig(fig)

    fig,ax = plt.subplots(1,1)
    nfreqs=0
    for ind, name in enumerate(keyswith("chspec_")):
        num = name.split("_")[1]
        spec = dat[name]
        nfreqs = len(spec)
        E = numpy.sum(spec**2)
        ax.plot(spec, label=f'{num} E={E:.2e}')
    ax.set_title("EmpiricalNoiseModel Spectra")
    ax.set_xlabel(f"frequency bins ({nfreqs})")
    plt.legend()
    out.savefig(fig)

