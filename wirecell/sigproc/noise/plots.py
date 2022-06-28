from wirecell import units
from wirecell.util.plottools import pages

import numpy
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt




def plot_many(spectra, outname):
    '''
    Make a multipage pdf with the spectral information.
    '''
    with pages(outname) as out:
        for ns in spectra:
            fig, ax = plt.subplots(nrows=1,ncols=1)
            freqs = numpy.asarray(ns["freqs"])
            freqs_mhz = freqs/units.MHz
            amps = numpy.asarray(ns["amps"])
            if amps is None:
                print(f'No spectrum!!! {list(ns.keys())}')
            amps_volt = amps/units.volt

            nsamples = ns["nsamples"]
            period_us = ns["period"]/units.us

            nsave = len(freqs)

            title = f'N={nsave}/{nsamples} T={period_us}us'
            if "plane" in ns:
                plane = ns["plane"]
                title += f' P={plane}'
            if "gain" in ns:
                gain_mvfc = ns["gain"]/(units.mV/units.fC)
                shaping_us = ns["shaping"]/units.us
                title += f' G={gain_mvfc}mV/fC S={shaping_us}us'
            if "wirelen" in ns:
                wl_cm = ns["wirelen"]/units.cm
                title += f' W={wl_cm}cm'
            g = ns.get("group", None) 
            if g is None:
                g = ns.get("groupID", None)
                if g is not None:
                    print("warning, please use 'group' instead of 'groupID")
            if g is not None:
                title += f' #{g}'

            ax.set_title(title)
            ax.set_xlabel('frequency [MHz]')
            ax.set_ylabel('amplitude [V]')
            ax.plot(freqs_mhz, amps_volt)
            out.savefig(fig)
            plt.close()
            
