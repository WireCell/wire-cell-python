from wirecell import units

import numpy
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def plot_many(spectra, pdffile):
    '''
    Make a multipage pdf with the spectral information
    '''
    with PdfPages(pdffile) as pdf:
        for ns in spectra:
            fig, ax = plt.subplots(nrows=1,ncols=1)
            
            freqs = numpy.asarray(ns.freqs)/units.MHz
            amps = numpy.asarray(ns.amps)/(units.volt/units.Hz)

            ax.set_title('plane=%d, wirelen=%.1f cm, gain=%.1f mV/fC shape=%.1f $\mu$s' % \
                             (ns.plane, ns.wirelen/units.cm, ns.gain/(units.mV/units.fC), ns.shaping/units.us))
            ax.set_xlabel('sampled frequency [MHz]')
            ax.set_ylabel('sampled amplitude [V/Hz]')
            ax.plot(freqs, amps)
            pdf.savefig(fig)
            plt.close()
            
