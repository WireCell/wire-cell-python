#!/usr/bin/env python
'''
Plots from data made by test_noisetools.  Use from wirecell-test.
'''


import re
import matplotlib.pyplot as plt
import numpy

def plot_waves(ax, waves, tit='white noise waves'):
    ax.set_title(tit)
    for wave in waves:
        ax.plot(wave)
    ax.set_xlabel('tick')


def wave_energies(waves):
    Es = list()
    for wave in waves:
        wsize = wave.size
        Es.append(numpy.sum(wave**2))
    return numpy.array(Es)    

def wave_rmses(waves):
    Es = wave_energies(waves)
    return numpy.sqrt(Es/waves[0].size)

def plot_hist_mean(ax, vals, tit='histogram'):
    ax.set_title(tit)
    ax.hist(vals, bins=50)
    xtr = ax.get_xaxis_transform()
    mu = numpy.sum(vals)/vals.size
    ax.vlines(mu, 0,1, transform=xtr, colors='y', label=f'mean {mu:.2f}')
    ax.legend()
    

def wave_plots(out, waves, name='white noise'):
    '''
    make plots from a set of waves
    '''
    wavesize = waves[0].size
    nwaves = len(waves)
    nex = 5

    fig,ax = plt.subplots(1,1)
    plot_waves(ax, waves[:nex], f'{name} waves ({nex} of {nwaves})')
    out.savefig(fig)
    plt.close(fig)

    Es = wave_energies(waves)
    fig,ax = plt.subplots(1,1)
    plot_hist_mean(ax, Es, f'{name} energies of {nwaves} waves of {Es.size} ticks')
    out.savefig(fig)
    plt.close(fig)

    rmses = wave_rmses(waves)
    fig,ax = plt.subplots(1,1)
    plot_hist_mean(ax, rmses, f'{name} RMSes of {nwaves} waves of {rmses.size} ticks')
    out.savefig(fig)
    plt.close(fig)

def plot_spec(ax, spec, name, lwid=1):
    lab = f'{name} N={spec.size}'
    ax.plot(numpy.linspace(0,2,spec.size), spec, linewidth=lwid, label=lab)
    ax.set_xlabel('frequency (per $F_{nyquist}$)')
    ax.legend()
    
def plot_specs(out, sname, tit, variants={}):
    fig, ax = plt.subplots(1,1)
    ax.set_title(tit)
    lwids=list(range(1, 1+len(variants)))
    lwids.reverse()
    for ind, (vname, spec) in enumerate(variants.items()):
        plot_spec(ax, spec, vname, lwids[ind])
    out.savefig(fig)
    plt.close(fig)

def plot_acs(out, tit, variants):
    lwids=list(range(1,1+len(variants)))
    lwids.reverse()

    # full
    fig,ax = plt.subplots(1,1)
    ax.set_title(tit)
    for lwid,(vname,ac) in zip(lwids,variants.items()):
        ac0 = ac[0]
        lab = f'{vname} N={ac.size} ac[0]={ac0:.1f}'
        ax.plot(ac, label=lab, linewidth=lwid);
    ax.set_xlabel('lag')
    ax.legend()
    out.savefig(fig)
    plt.close(fig)

    # half
    fig,ax = plt.subplots(1,1)
    ax.set_title(tit + " (half)")
    for lwid,(vname,ac) in zip(lwids,variants.items()):
        ac0 = ac[0]
        lab = f'{vname} N={ac.size} ac[0]={ac0:.1f}'
        ax.plot(ac[:ac.size//2], label=lab, linewidth=lwid);
    ax.set_xlabel('lag')
    ax.legend()
    out.savefig(fig)
    plt.close(fig)
    
class NamArr:
    def __init__(self, name, arr):
        parts = name.split('_');
        # print(parts)
        self.arr = arr
        self.kind = parts[0]    # eg "amp", "trusig"
        self.proto = parts[1]   # eg "white" or "shape"
        self.cycle = parts[2] == "c1";
        self.nticks = int(parts[3][1:])
        self.trip = int(parts[4][1:])
        if self.kind == 'wav':
            self.index = int(parts[5])
    

class Dat:
    def __init__(self, dat, isnamarr=False):
        if isnamarr:
            self.dat = dat
        else:
            self.dat = {k:NamArr(k,a) for k,a in dat.items()}

    def __call__(self, arrs=False, **query):
        '''
        Query the data.
        '''
        d = dict(self.dat)
        for k,q in query.items():
            if k == "name":
                qre = re.compile(q)
                d = {n:na for n,na in d.items() if qre.match(n)}
            if type(q) == int:
                d = {n:na for n,na in d.items() if q == getattr(na, k, None)}
            else:               # regex on NamArr.<k>
                qre = re.compile(q)
                d = {n:na for n,na in d.items() if qre.match(getattr(na, k, None))}
        if arrs:
            return [v.arr for v in d.values()]
        return Dat(d, True)

    def gots(self, cat):
        '''
        Return list of values of category.
        '''
        ret = set()
        for one in self.dat.values():
            ret.add(getattr(one, cat))
        ret = list(ret)
        ret.sort()
        return ret
        
def plot_proto(dat, name, out):
    '''
    Make plots for proto of name.
    '''
    specnames = ('sig', 'amp', 'lin', 'sqr', 'rms', 'per', 'psd')

    waves = dat(kind='wav')
    assert(len(waves.dat)>0)
    for cycle in [0,1]:
        for trip in [1,2]:
            ws = waves(arrs=True, cycle=cycle, trip=trip)
            wave_plots(out, ws, f'{name} waves c{cycle} r{trip}')

    for specname in specnames:
        dspecs = dat(kind=specname)
        fig, ax = plt.subplots(1,1)
        ax.set_title(f'{name} {specname}')

        for cycle in [0,1]:
            for trip in [1,2]:
                d = dspecs(arrs=True, cycle=cycle, trip=trip)
                
                t = dat(arrs=True, kind='tru'+specname, cycle=cycle, trip=trip)
                if len(t):
                    t = t[0]
                    lab = f'c{cycle} r{trip} [true]'
                    ax.plot(numpy.linspace(0,2,t.size), t, linewidth=0.5, label=lab)                    
  

                assert(len(d) == 1)
                spec = d[0]
                lab = f'c{cycle} r{trip} [{spec.size}]'
                ax.plot(numpy.linspace(0,2,spec.size), spec, linewidth=1, label=lab)
                ax.set_xlabel('frequency (per $F_{nyquist}$)')
        ax.legend()
        out.savefig(fig)
        plt.close(fig)

    bacs = dat(kind='bac')
    sacs = dat(kind='sac')

    dbacs=dict()
    dsacs=dict()

    for cycle in [0,1]:
        for trip in [1,2]:
            dbacs[f'c{cycle} r{trip}'] = bacs(arrs=True, cycle=cycle, trip=trip)[0]
            dsacs[f'c{cycle} r{trip}'] = sacs(arrs=True, cycle=cycle, trip=trip)[0]


    plot_acs(out, f'{name} biased autocorrelation', dbacs)
    plot_acs(out, f'{name} unbiased sample autocorrelation', dsacs)

def plot(dat, out):
    '''
    Make plots for the test_spectra output.  
    '''
    dat = Dat(dat)

    for proto in ("white", "gauss", "shape"):
        wdat = dat(proto=proto)
        plot_proto(wdat, proto, out);

    # plot_proto(sdat, out, 'round', ('cyclic128r', 'acyclic128r'))


def plot_junk(dat, out):
    dat = Dat(dat)

    plot_collecting(dat, out)

    # Categories
    autocorrs = ('bac', 'sac')
    specnames = ('meanlinear', 'rootmeansquare', 'meansquare', 'periodogram', 'psd')

    variants = ['cyclic7', 'acyclic7']
    gen_variants = ['block', 'normed', 'interp']
    
    waves = list(dat.like("wave",vname).values())
    nwaves = len(waves)

    wave_plots(out, waves, 'white noise')

    lwids = [3, 1]

    variants.reverse()
    for sname in specnames:
        plot_specs(out, sname, f'white noise {sname}', {n:dat.get(sname, n) for n in variants})

    plot_acs(out, 'White noise biased autocorrelation', {
        v:dat.get('bac', v) for v in variants})
    plot_acs(out, 'White noise unbiased sample autocorrelation', {
        v:dat.get('sac', v) for v in variants})


    ### now plots for generated 

    for vname in gen_variants:  # block/interp
        waves = list(dat.like('wave', vname).values())
        wave_plots(out, waves, f'{vname}')

    for sname in specnames:
        plot_specs(out, sname, f'{vname} {sname} spectra',
                   {sname:dat.get(sname, f'{vname}_roundtrip') for vname in gen_variants})

    plot_acs(out, 'Roundtrip biased autocorrelation', {
        v:dat.get('bac', f'{v}_roundtrip') for v in gen_variants})
    plot_acs(out, 'Roundtrip unbiased sample autocorrelation', {
        v:dat.get('sac', f'{v}_roundtrip') for v in gen_variants})

    cols=dict(truespec='k', meanspec='b', mrmsspec='r')
    def one_spec(ax, vname, key):
        s = dat.get(vname, key)
        med = numpy.median(s)
        c = cols[key]
        sname = key[:-4]         # remove "spec"
        ytr = ax.get_yaxis_transform()
        ax.hlines(med, 0,1, transform=ytr, linewidth=.1, colors=c)
        freqs = numpy.linspace(0,2,s.size)
        lwid=1
        lab = f'{med:.1f} {vname} {sname} spec'
        if vname == 'interp':
            lwid=0.5
        ax.plot(freqs, s, label=lab, linewidth=lwid, c=c)

    for vname in gen_variants:
        fig,ax = plt.subplots(1,1)
        ax.set_title(f'{vname} roundtrip spectrum')
        for one in cols:
            one_spec(ax, vname, one)
        ax.legend()
        out.savefig(fig)
        plt.close(fig)
    
