import math
import click
from wirecell import units
from wirecell.util.functions import unitify, unitify_parse
from wirecell.util import ario
import numpy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

@click.group("test")
@click.pass_context
def cli(ctx):
    '''
    Wire Cell Test Commands
    '''

@cli.command("noise")
@click.argument("ario-file")
@click.argument("pdf-file")
@click.pass_context
def noise(ctx, ario_file, pdf_file):
    '''
    Process test_noise.tar
    '''
    fp = ario.load(ario_file)
    
    ss_freq = fp["ss_freq"]
    ss_spec = fp["ss_spec"]

    freqs = fp["freqs"]
    spec = fp["true_spectrum"]

    nexample = 100;

    cats = ("fresh", "recycled", "oldnoise")

    with PdfPages(pdf_file) as pdf:

        fig,ax = plt.subplots(1,1)
        ax.plot(freqs, spec, label="regular")
        ax.plot(ss_freq, ss_spec, label="irregular")
        ax.set_title("irregular and regular sampled 'true' spectra")
        plt.legend()
        pdf.savefig(fig)
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
            pdf.savefig(fig)
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
            pdf.savefig(fig)

            aspec = fp[f"{name}_spectrum"]
            specs.append(aspec)
            fig,ax = plt.subplots(1,1)
            ax.plot(freqs, aspec, label=name)
            ax.plot(freqs, spec, label="true")
            ax.set_title(f'True and "{name}" spectra')
            fig.legend()
            pdf.savefig(fig)

        fig,ax = plt.subplots(1,1)
        ax.set_title("difference from true")
        n = len(freqs)//2
        for ind, name in enumerate(cats):
            ax.plot(freqs[:n], (specs[ind]-spec)[:n], label=name)
        fig.legend()
        pdf.savefig(fig)

        plt.close()





def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
