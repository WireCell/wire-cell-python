#!/usr/bin/env python
'''
Work with sim data.

fixme: this module is poorly named.
'''

from wirecell import units
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.rcParams['axes.formatter.useoffset'] = False

def baseline_subtract(frame):
    '''
    Return a new frame array (chan x tick) were each channel row has
    its baseline subtracted.
    '''
    # gotta transpose around to get the right shapes to trigger broadcasting.
    return (frame.T - numpy.median(frame, axis=1)).T


def parse_channel_boundaries(cb):
    if type(cb) != type(""):
        return cb               # assume already parsed.
    return tuple(map(int, cb.split(',')))
    

def group_channel_indices(channels, boundaries=()):
    '''
    Given a list of channels, return a list of lists where each
    sublist is a sequential set.  If a sequence of boundaries are
    given, then force a grouping on that channel even if there's not
    jump.
    '''

    print ("Channel boundaries: %s" % str(boundaries))

    channels = numpy.asarray(channels)
    chan_list = channels.tolist()
    binds = list()
    for b in boundaries:
        try:
            i = chan_list.index(b)
        except ValueError:
            continue
        binds.append(i)

    chd = channels[1:] - channels[:-1]
    chjumps = [-1] + list(numpy.where(chd>1)[0]) + [channels.size-1]
    ret = list()
    for a,b in zip(chjumps[:-1], chjumps[1:]):
        #print a,b,channels[a+1:b+1]
        a += 1
        b += 1

        gotsome = 0
        for bind in binds:
            if a < bind and bind < b:
                ret.append((a,bind))
                ret.append((bind,b))
                gotsome += 1
        if gotsome == 0:
            ret.append((a,b))
    return ret

class Frame(object):
    def __init__(self, fp, ident=0, tag=''):
        def thing(n):
            arr = fp['%s_%s_%d' % (n, tag, ident)]
            if n == 'frame':
                return arr.T
            return arr

        for n in 'frame channels tickinfo'.split():
            setattr(self, n, thing(n))
        
        self.channel_boundaries = ()

    def plot_ticks(self, tick0=0, tickf=-1, raw=True, chinds = ()):
        '''
        Plot in terms of ticks.  Here, the frame is assumed to be
        dense and chinds are taken as channel ranges.
        '''
        frame = self.frame
        print ("Frame shape: %s" % str(frame.shape))

        if not raw:
            frame = baseline_subtract(frame)


        if not chinds:
            chinds = group_channel_indices(self.channels, self.channel_boundaries)

        tick = self.tickinfo[1]
        if tickf < 0:
            tickf += frame.shape[1]

        ngroups = len(chinds)
        fig, axes = plt.subplots(nrows=ngroups, ncols=1, sharex = True)
        if ngroups == 1:
            axes = [axes]       # always list 

        for ax, chind in zip(axes, chinds):
            ngroups -= 1
            ch1 = self.channels[chind[0]]
            ch2 = self.channels[chind[1]-1]

            extent = (tick0, tickf, ch2, ch1)
            print ("exent: %s" % str(extent))

            im = ax.imshow(frame[chind[0]:chind[1],tick0:tickf],
                           aspect='auto', extent=extent, interpolation='none')
            plt.colorbar(im, ax=ax)
            if not ngroups:
                ax.set_xlabel('ticks (%.2f us)' % (tick/units.us),)

        return fig,axes

    def plot(self, t0=None, tf=None, raw=True, chinds = ()):

        frame = self.frame
        if not raw:
            frame = baseline_subtract(frame)

        tstart, tick = self.tickinfo[:2]
        nticks = frame.shape[1]
        tend = tstart + nticks*tick

        if t0 is None or t0 < tstart or t0 > tend:
            t0 = tstart

        if tf is None or tf < t0 or tf > tend:
            tf = tend

        tick0 = int((t0-tstart)/tick)
        tickf = int((tf-tstart)/tick)
        
        #print ("trange=[%.2f %.2f]ms ticks=[%d,%d]" % (t0/units.ms,tf/units.ms,tick0,tickf))

        if not chinds:
            chinds = group_channel_indices(self.channels)
        ngroups = len(chinds)
        fig, axes = plt.subplots(nrows=ngroups, ncols=1, sharex = True)
        if ngroups == 1:
            axes = [axes]       # always list 

        for ax, chind in zip(axes, chinds):
            ngroups -= 1
            chind0 = chind[0]
            chind1 = chind[1]

            ch1 = self.channels[chind0]
            ch2 = self.channels[chind1-1]

            extent = (t0/units.ms, tf/units.ms, ch2, ch1)
            print ("exent: %s, chind=(%d,%d)" % (str(extent), chind0, chind1) )


            #print (chind,extent, chind, tick0, tickf, frame.shape)
            im = ax.imshow(frame[chind0:chind1,tick0:tickf],
                           aspect='auto', extent=extent, interpolation='none')
            plt.colorbar(im, ax=ax)
            if not ngroups:
                ax.set_xlabel('time [ms]')

        #plt.savefig(outfile)
        return fig,axes
        
        #plt.imshow(fr[ch[0]:ch[1], ti[0]:ti[1]], extent=[ti[0], ti[1], ch[1], ch[0]], aspect='auto', interpolation='none'); plt.colorbar()


# f = numpy.load("test-boundaries.npz"); reload(sim);
# fo = sim.Frame(f)
# fo.plot()
# dd = f['depo_data_0']
# plt.scatter(dd[2]/units.m, dd[4]/units.m)

class Depos(object):
    def __init__(self, fp, ident=0):
        # 7xN: t, q, x, y, z, dlong, dtran,
        self.data = fp['depo_data_%d'%ident]
        # 4xN: id, pdg, gen, child
        self.info = fp['depo_info_%d'%ident]

    @property
    def t(self):
        return self.data[0]
    @property
    def q(self):
        return self.data[1]
    @property
    def x(self):
        return self.data[2]
    @property
    def y(self):
        return self.data[3]
    @property
    def z(self):
        return self.data[4]

    def plot(self):
        fig = plt.figure()
        ax10 = fig.add_subplot(223)
        ax00 = fig.add_subplot(221)
        ax11 = fig.add_subplot(224, sharey=ax10)
        ax01 = fig.add_subplot(222, projection='3d')

        
        ax00.hist(self.t/units.ms, 100)

        ax10.scatter(self.x/units.m, self.z/units.m)
        ax11.scatter(self.y/units.m, self.z/units.m)
        ax01.scatter(self.x/units.m, self.y/units.m, self.z/units.m)

        ax00.set_title('depo times [ms]');

        ax10.set_xlabel('x [m]');
        ax10.set_ylabel('z [m]');
        ax11.set_xlabel('y [m]');
        ax11.set_ylabel('z [m]');


        # fig, axes = plt.subplots(nrows=2, ncols=2)
        # axes[0,0].scatter(self.x/units.m, self.y/units.m, sharex=axes[1,0])
        # axes[1,0].scatter(self.x/units.m, self.z/units.m)
        # axes[1,1].scatter(self.y/units.m, self.z/units.m, sharey=axes[1,0])
        # axes[0,1].scatter(self.x/units.m, self.y/units.m, self.z/units.m)
        return fig,(ax00,ax10,ax01,ax11)

class NumpySaver(object):
    def __init__(self, filename):
        self.fname = filename
        self.reload()

    def reload(self):
        f = numpy.load(filename)
        
