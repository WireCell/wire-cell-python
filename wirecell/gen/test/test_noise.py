from wirecell.gen.noise import *
from math import log2
import numpy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams['figure.constrained_layout.use'] = True

def spec_wids(specs):
    sizes = numpy.array([s.size for s in specs])
    m = log2(sizes.min())
    wids = 0.5 * (1 + numpy.log2(sizes) - m)
    return wids


class TestNoise:

    def __init__(self, pdf, nsamples=256, period=1.0, rms=1.0, nrm=0.2): 
        self.pdf = pdf
        self.nsamples = nsamples
        self.smaller_nsamples = (nsamples//4, nsamples//3, nsamples//2, int(nsamples*0.9), nsamples)
        self.larger_nsamples = (nsamples, int(nsamples*1.1), int(nsamples*2), int(nsamples*3), int(nsamples*4))
        self.other_nsamples = list(set(self.smaller_nsamples + self.larger_nsamples))
        self.other_nsamples.sort()

        self.rms = rms          # a time-series RMS
        self.nrm = nrm          # where the fictional spectrum mode is.
        self.period = period
        self.other_periods = (0.5*period, 0.9*period, period, 1.1*period, 2*period)

    def save(self):
        self.pdf.savefig(plt.gcf())
        plt.close();

    def plot_spec(self, ax, spec, label=None, wid=1):
        ax.plot(spec.freqs, spec.amp, label=label, linewidth=wid)
        ax.set_xlabel('frequency')
        ax.set_ylabel('amplitude')

    def white(self, nwaves=None):
        if not nwaves:
            nwaves = self.nsamples
        fig,(ax1,ax2) = plt.subplots(2,1)
        
        gwaves = gaussian_waves(self.rms, self.nsamples, nwaves)
        gene = waves_energy(gwaves)
        grms = waves_rms(gwaves)
        for w in gwaves[:5]:
            ax1.plot(w)
        ax1.set_title(f'white RMS={self.rms:.1f} Ew={gene:.3f} RMSw={grms:.3f}')

        col = Collect(self.nsamples)
        for gwave in gwaves:
            col.add(gwave)
        cene = col.energy
        spec = Spec(col.linear, self.period)
        sene = spec.energy
        self.plot_spec(ax2, spec, f'col: E={sene:.3f}')

        rt = spec.roundtrip()
        rene = rt.energy
        self.plot_spec(ax2, rt, f'rt: E={rene:.3f}')
        ax2.set_title('white spectrum')
        ax2.legend()

        #print(f'RMS={grms:.3f}, gE={gene:.1f} cE={cene:.1f} sE={sene:.1f} rE={rene:.1f}')
        self.save()

    def interp(self):
        'nrm = noise relative mode' 
        newsizes = self.other_nsamples
        freqs = frequencies(self.nsamples, self.period)
        tru = hermitian_mirror(rayleigh(freqs, self.nrm))
        spec = Spec(tru, self.period)
        sene = spec.energy

        specs = [spec]
        specs += [spec.interp(ns) for ns in newsizes]

        specs.reverse()
        wids = spec_wids(specs)

        fig,ax = plt.subplots(1,1)
        for s,w in zip(specs,wids):
            waves = s.waves()
            lab=f'{s.size:4d}: Es={s.energy:4.1f} Ew={waves_energy(waves):4.1f}, RMSw={waves_rms(waves):.3f}'
            self.plot_spec(ax, s, lab, w)
        ax.set_title(f'interpolate from {self.nsamples}, T={self.period} nrm={self.nrm}')
        ax.legend()
        self.save()


    def extrap(self):
        newsizes = self.larger_nsamples;
        freqs = frequencies(self.nsamples, self.period)
        tru = hermitian_mirror(rayleigh(freqs, self.nrm))
        spec = Spec(tru, self.period)
        sene = spec.energy

        specs = [spec]
        specs += [spec.extrap(ns, 0.0) for ns in newsizes]
        specs.reverse()
        wids = spec_wids(specs)

        fig,ax = plt.subplots(1,1)
        for s,w in zip(specs,wids):
            waves = s.waves()
            tt = s.period*s.size
            se = s.energy
            we = waves_energy(waves)
            wrms = waves_rms(waves)
            lab = f'{s.size}: tt={tt:.0f} sE={se:.1f} wE={we:.1f} RMS={wrms:.3f}'
            self.plot_spec(ax, s, lab, w)
        ax.set_title(f'extrapolate from {self.nsamples}, T={self.period} nrm={self.nrm}')
        ax.legend()
        self.save()

    def alias(self, nhalfs=10):
        nhalfs = int(log2(self.nsamples))
        freqs = frequencies(self.nsamples, self.period)
        tru = hermitian_mirror(rayleigh(freqs, self.nrm))
        spec = Spec(tru, self.period)
        sene = spec.energy

        specs = [spec]
        while nhalfs > 4:
            nhalfs -= 1
            ow = specs[-1]
            ns = 1 << nhalfs
            nspec = ow.alias(ns)
            specs.append(nspec)

        wids = spec_wids(specs)
        fig,ax = plt.subplots(1,1)
        more = list()
        for s,w in zip(specs,wids):
            waves = s.waves()
            tt = s.period
            se = s.energy
            we = waves_energy(waves)
            wrms = waves_rms(waves)
            lab=f'{s.size:4d}: T={tt:.0f} sE={se:.1f} wE={we:.1f} RMS={wrms:.3f}'
            more.append((lab, waves, s))
            self.plot_spec(ax, s, lab, w)
        ax.set_title(f'alias from {self.nsamples}, T={self.period} nrm={self.nrm}')
        ax.legend()
        self.save()

        for tit, waves, spec in more:
            fig,(ax1,ax2) = plt.subplots(2,1)
            ax1.set_title('alias: '+tit)
            self.plot_spec(ax1, spec)
            for wave in waves[:5]:
                ax2.plot(wave)
            ax2.set_xlabel("time sample")
            ax2.set_title("waves")
            self.save()


    def resample(self):
        sizes = self.other_nsamples
        periods = self.other_periods
        freqs = frequencies(self.nsamples, self.period)
        tru = hermitian_mirror(rayleigh(freqs, self.nrm))
        spec = Spec(tru, self.period)

        se = spec.energy
        waves = spec.waves()
        we = waves_energy(waves)
        wrms = waves_rms(waves)


        for siz in sizes:
            for per in periods:
                try:
                    rs = spec.resample(siz,per)
                except ValueError as e:
                    # print(e)
                    # print(f'{siz:4d}: per={per:.2f} can not resample')
                    continue
                tit = f'resample N={spec.size:4d}, T={spec.period:.1f} to N={siz}, T={per}'
                fig, ax = plt.subplots(1,1)
                ax.set_title(tit)

                waves = rs.waves()
                rse = rs.energy
                rwe = waves_energy(waves)
                rwrms = waves_rms(waves)

                self.plot_spec(ax, spec,
                               f'N={spec.size} T={spec.period:.1f} Es={se:4.1f} Ew={we:4.1f} RMS={wrms:.1f}')
                self.plot_spec(ax, rs,
                               f'N={rs.size} T={rs.period:.1f} Es={rse:4.1f} Ew={rwe:4.1f} RMS={rwrms:.1f}')
                ax.legend()
                self.save()

    def downsample_white(self):
        nhalfs = int(log2(self.nsamples))
        waves = [gaussian_waves(self.rms, self.nsamples)]
        while nhalfs > 4:
            nhalfs -= 1
            ow = waves[-1]
            half = ow.shape[1]//2
            ws = ow[:,:half] + ow[:,half:]
            waves.append(ws)

        more = list()
        for ws in waves:
            col = Collect(ws[0].size)
            for w in ws:
                col.add(w)
            spec = Spec(col.linear, ws[0].size/self.nsamples)
            rms = waves_rms(ws)
            more.append((spec,rms))

        wids = spec_wids([m[0] for m in more]).tolist()
        more.reverse()

        fig,ax = plt.subplots(1,1)
        ax.set_title(f'downsampling white from {self.nsamples}')
        for (spec,rms),wid in zip(more,wids):
            E = spec.energy
            lab = f'{spec.size:4d}: E={E:.1f} RMS={rms:.3f}'
            self.plot_spec(ax, spec, lab, wid)
        ax.legend()
        self.save()

def doit(nsamples=256, period=1.0, rms=1.0, nrm=0.2):
    rms100 = int(100*rms)
    p100 = int(100*period)
    nrm100 = int(100*nrm)
    filename = f'test_noise_N{nsamples}_T{p100}_sigma{nrm100}_rms{rms100}.pdf'
    print(filename)
    with PdfPages(filename) as pdf:
        tn = TestNoise(pdf, nsamples, period, rms, nrm)
        tn.white()
        tn.interp()
        tn.extrap()
        tn.alias()
        tn.resample()
        tn.downsample_white()
        
if '__main__' == __name__:
    doit(nrm=0.2)
    doit(nrm=0.1)
    
