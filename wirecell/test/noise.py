import numpy
import matplotlib.pyplot as plt

def plot(fp, out):
    '''
    Make plots from data produced by test_noise.

    The "fp" must be a dictionary of arrays as loaded from the output
    file produced by test_noise.

    The "out" must be an object like
    wirecell.util.plottools.NameSequence.
    '''

    ss_freq = fp["ss_freq"]
    ss_spec = fp["ss_spec"]

    freqs = fp["freqs"]
    spec = fp["true_spectrum"]

    nexample = 100;

    cats = ("fresh", "recycled", "oldnoise")



    fig,ax = plt.subplots(1,1)
    ax.plot(freqs, spec, label="regular")
    ax.plot(ss_freq, ss_spec, label="irregular")
    ax.set_title("irregular and regular sampled 'true' spectra")
    plt.legend()
    out.savefig(fig)
    plt.close()

    inrange = fp["inrange"]
    fig,ax = plt.subplots(1,1)
    ax.hist(inrange, bins=numpy.max(inrange)+1)
    ax.set_title("range samples")
    out.savefig(fig)
    plt.close()

    specs = list()
    for name in cats:
        fig,ax = plt.subplots(1,1)
        for ind in range(5):
            w = fp[f'{name}_wave{ind:03d}']
            e = numpy.sum(w*w)
            plt.plot(w, label=f'{e:.1f}')
        plt.legend();
        ax.set_title(f'Example "{name}" waves')
        out.savefig(fig)
        plt.close()

        fig,ax = plt.subplots(1,1)
        energies = list()
        for ind in range(nexample):
            w = fp[f'{name}_wave{ind:03d}']
            e = numpy.sum(w*w)
            energies.append(e)
        print(f'energies {name}:', numpy.sum(energies)/len(energies))
        plt.hist(energies)
        ax.set_title(f'Energies of "{name}" waves')
        out.savefig(fig)

        if name in ("recycled",):
            fig,ax = plt.subplots(2,1)
            energies = list()
            for ind in range(5):
                rw = fp[f'{name}_real{ind:03d}']
                iw = fp[f'{name}_imag{ind:03d}']
                ax[0].plot(rw)
                ax[1].plot(iw)
                print(ind, rw[0], iw[0])
            ax[0].set_title(f'Real spectra of "{name}" waves')
            ax[1].set_title(f'Imginary spectra of "{name}" waves')
            out.savefig(fig)


        aspec = fp[f"{name}_spectrum"]
        specs.append(aspec)
        fig,ax = plt.subplots(1,1)
        ax.plot(freqs, aspec, label=name)
        ax.plot(freqs, spec, label="true")
        ax.set_title(f'True and "{name}" spectra')
        fig.legend()
        out.savefig(fig)

    fig,ax = plt.subplots(1,1)
    ax.set_title("difference from true")
    n = len(freqs)//2
    for ind, name in enumerate(cats):
        ax.plot(freqs[:n], (specs[ind]-spec)[:n], label=name)
    fig.legend()
    out.savefig(fig)
    plt.close()

