#!/usr/bin/env python

import os
import sys
import click

from wirecell import units

@click.group("validate")
@click.pass_context
def cli(ctx):
    '''
    Wire Cell Validation
    '''
    pass


@cli.command("diff-hists")
@click.option("-n", "--name", multiple=True,
              help="Name of histogram in each file")
@click.option("-o", "--out", default=None,
              help="Output file")
@click.argument("file1")
@click.argument("file2")
@click.pass_context
def diff_hists(ctx, name, out, file1, file2):
    '''
    Produce an output ROOT file which  holds the difference of a histogram from two files.
    '''

    import root
    tfile1 = root.open_file(file1)
    tfile2 = root.open_file(file2)
    if out:
        out = root.open_file(out, "recreate")

    tosave=list()
    for one in name:
        one = str(one)          # unicode messes up ROOT
        h1 = root.load_obj(tfile1, one)
        h2 = root.load_obj(tfile2, one)
        h1.Add(h2, -1.0)
        if out:
            h1.SetDirectory(out)
            tosave.append(h1)
    if not out:
        return
    out.Write()
    out.Close()


@cli.command("magnify-diff")
@click.option("-e", "--epsilon", default=0.0,
              help="The maximum delta for two histograms to be considered different")
@click.option("-o", "--out", 
              help="Output file")

@click.argument("file1")
@click.argument("file2")
@click.pass_context
def magnify_diff(ctx, epsilon, out, file1, file2):
    '''
    Form a new Magnify file holds histograms which are the difference of those from the two inputs.
    '''
    if not out:
        sys.exit(1)

    from time import time
    import root
    tfile1 = root.open_file(file1)
    tfile2 = root.open_file(file2)
    out = root.open_file(out, "recreate")
    
    names1 = set([key.GetName() for key in tfile1.GetListOfKeys() if root.is_hist(key)])
    names2 = set([key.GetName() for key in tfile2.GetListOfKeys() if root.is_hist(key)])

    loners = names1 ^ names2
    for loner in loners:
        print "loner: %s" % loner

    names = list(names1 & names2)
    names.sort()
    t1 = time()
    hists1 = [tfile1.Get(name) for name in names]
    hists2 = [tfile2.Get(name) for name in names]
    t2 = time()
    #print "load: ", t2-t1
    for name,obj1,obj2 in zip(names, hists1, hists2):
        obj1.Add(obj2, -1.0)
        mi = obj1.GetMinimum()
        ma = obj1.GetMaximum()
        if abs(mi) > epsilon or abs(ma) > epsilon:
            msg = "diff: %s %e %e" % (name, mi, ma)
            print msg
        obj1.SetDirectory(out)
    t3 = time()
    #print "subtract: ", t3-t2
    out.Write()
    out.Close()
    t4 = time()
    #print "done: ", t4-t3

@cli.command("magnify-jsonify")
@click.option("-o", "--out", 
              help="Output file")
@click.argument("filename")
@click.pass_context
def magnify_jsonify(ctx, out, filename):
    '''
    Jsonnify summary info about all histograms.
    '''
    import json
    import root
    tfile = root.open_file(filename)
    names = list()

    dat = dict();
    for key in tfile.GetListOfKeys():
        if key.IsFolder():
            continue
        if not root.is_hist(key):
            continue
        d = root.hist_to_dict(key.ReadObj());
        name = key.GetName()
        dat[name] = d
        #print name, d
    open(out,"w").write(json.dumps(dat, indent=4))

@cli.command("magnify-dump")
@click.option("-o", "--out", 
              help="Output file")
@click.argument("filename")
@click.pass_context
def magnify_dump(ctx, out, filename):
    '''Dump magnify histograms into Numpy .npz files.

    The underlying histogram array is dumped into an array of the same
    name.  For every dimension of histogram an array of bin edges is
    dumped as <name>_<dim>edges where <dim> is "x", "y" or "z".
    '''
    import ROOT
    import numpy
    from root_numpy import hist2array
    from time import time

    print "Reading ROOT file"
    t1 = time()
    arrs = dict()
    tfile = ROOT.TFile.Open(filename)
    for key in tfile.GetListOfKeys():
        cname = key.GetClassName()
        if cname[:3] not in ["TH1", "TH2", "TH3" ]:
            continue

        th = key.ReadObj()
        hname = th.GetName()
        arr,edges = hist2array(th,return_edges=True)
        arrs[hname] = arr
        for dim,edge in enumerate(edges):
            arrs["%s_%sedge"%(hname, "xyz"[dim])] = edge
    t2 = time()
    print t2-t1                 # takes 5.7 seconds compared to 5.3 seconds loading npz
    print "Writing NPZ file"
    numpy.savez_compressed(out, **arrs)
    t3 = time()
    print t3-t2
        
@cli.command("npz-load")
@click.option("-o", "--out", 
              help="Output file")
@click.argument("filename")
@click.pass_context
def npz_load(ctx, out, filename):
    import numpy
    from time import time
    t1 = time()
    arrs1 = numpy.load(filename,'r+')
    t2 = time()
    print 'Memmap load: %f'%(t2-t1,)
    arrs2 = numpy.load(filename,None)
    t3 = time()
    print 'Full load: %f'%(t3-t2)
    outs=dict()
    for name in arrs1:
        arr1 = arrs1[name]
        arr2 = arrs2[name]
        arr = arr2-arr1
        outs[name] = arr
    t4 = time()
    print 'Subtract: %f' % (t4-t3)
    numpy.savez_compressed(out, **outs)
    t5 = time()
    print 'Done: %f' % (t5-t4)

@cli.command("magnify-plot-reduce")
@click.option("-n", "--name", default="orig",
              help="The histogram type name")
@click.option("-o", "--out", 
              help="Output file")
@click.argument("filename")
@click.pass_context
def magnify_plot_reduce(ctx, name, out, filename):
    '''
    Reduce a magnify 2D histogram along the time dim.
    '''
    import ROOT
    import numpy
    from root_numpy import hist2array
    from wirecell.validate.plots import channel_summaries as plotter

    methods = [
        ("min", numpy.min),
        ("max", numpy.max),
        ("sum", numpy.sum),
        ("rms", lambda a: numpy.sqrt(numpy.mean(numpy.square(a)))),
        ("absmin", lambda a: numpy.min(numpy.abs(a))),
        ("absmax", lambda a: numpy.max(numpy.abs(a))),
        ("abssum", lambda a: numpy.sum(numpy.abs(a))),
    ]
    name = str(name)
    tfile = ROOT.TFile.Open(filename)
    hists = [tfile.Get("h%s_%s"%(letter, name)) for letter in "uvw"]
    aes = [hist2array(hist, return_edges=True) for hist in hists]

    edges = [ae[1] for ae in aes]
    extents = [(e[0][0],e[0][-1]) for e in edges]
    #print extents

    arrs_by_name = dict()
    for mname, meth in methods:
        arrs_by_name[mname] = [numpy.apply_along_axis(meth, 1, ae[0]) for ae in aes]

    fig = plotter(name, arrs_by_name, extents)
    #fig.suptitle('Channel Summaries for "%s"' % name)
    fig.savefig(out)
    

@cli.command("magnify-plot")
@click.option("-n", "--name", default="orig",
              help="The histogram type name")
@click.option("-t", "--trebin", type=int, default=1,
              help="Set amount of rebinning in time domain (must be integral factor)")
@click.option("-c", "--crebin", type=int, default=1,
              help="Set amount of rebinning in channel comain (must be integral factor)")
@click.option("--baseline/--no-baseline", default=False, 
              help="Calculate, subtract and display baselines.")
@click.option("--threshold", type=float, default=0.0,
              help="Apply a threshold.")
@click.option("--saturate", type=float, default=0.0,
              help="Saturate values.")
@click.option("-o", "--out", 
              help="Output file")
@click.argument("filename")
@click.pass_context
def magnify_plot(ctx, name, trebin, crebin, baseline, threshold, saturate, out, filename):
    '''
    Plot magnify histograms.
    '''
    import numpy
    from root_numpy import hist2array as h2a
    from wirecell.validate.arrays import rebin, bin_ndarray
    from wirecell.validate.plots import three_horiz
    import ROOT 

    name = str(name)

    tfile = ROOT.TFile.Open(filename)
    hists = [tfile.Get("h%s_%s"%(letter, name)) for letter in "uvw"]
    if not all(hists):
        raise ValueError('Could not get hists for "%s" from: %s' % (name, filename))

    # loading from ROOT takes the most time.

    aes = [h2a(h,return_edges=True) for h in hists]
    arrs = list()
    extents = list()
    for h,(a,e) in zip(hists,aes):
        xa,ya = [getattr(h,"Get"+l+"axis")() for l in "XY"]
        nx,ny = [getattr(h,"GetNbins"+l)() for l in "XY"]
        
        # h2a returns tick-major shape (nchan,ntick).
        ce,te = e
        ext = ((ce[0],ce[-1]), (te[0],te[-1]))

        print "%s: X:%d in [%.0f %.0f] Y:%d in [%.0f %.0f]" % \
            (h.GetName(),
             nx, xa.GetBinLowEdge(1), xa.GetBinUpEdge(nx),
             ny, ya.GetBinLowEdge(1), ya.GetBinUpEdge(ny),)
        print "\t shape=%s ext=%s" % (a.shape, ext)

        arrs.append(numpy.fliplr(a))
        extents.append(ext)


    norm = 1.0/(crebin*trebin)


    baselines = list()
    if baseline:
        newarrs = list()
        for arr in arrs:
            newarr = list()
            bl = list()         # across channels in one plane
            for wav in arr:
                digi = numpy.int32(wav)
                #histo = numpy.bincount(digi)
                #val = numpy.argmax(histo)
                values, counts = numpy.unique(digi, return_counts=True)
                ind = numpy.argmax(counts)
                val = values[ind]
                bl.append(val)
                wav = wav - val
                newarr.append(wav)
            newarr = numpy.asarray(newarr)
            bla = numpy.asarray(bl)
            bla = rebin(bla, bla.size/crebin)
            baselines.append(bla)
            newarrs.append(newarr)
        arrs = newarrs
    #arrs = [norm*rebin(arr, arr.shape[0]/crebin, arr.shape[1]/trebin) for arr in arrs]
    arrs = [bin_ndarray(arr, (arr.shape[0]/crebin, arr.shape[1]/trebin), "mean") for arr in arrs]

    tit = 'Type "%s" (rebin: ch=x%d tick=x%d), file:%s' % (name, crebin, trebin, os.path.basename(filename))
    
    if threshold > 0.0:
        #print "thresholding at", threshold
        tit += " [threshold at %d]" % threshold
        arrs = [numpy.ma.masked_where(numpy.abs(arr)<=threshold, arr) for arr in arrs]

    if saturate > 0.0:
        #print "saturating at", saturate
        #tit += "saturate=%d" % saturate
        newarrs = list()
        for arr in arrs:
            arr[arr > saturate] =  saturate
            arr[arr <-saturate] = -saturate
            newarrs.append(arr)
        arrs = newarrs

    # switch from tick-major (nchans, nticks) to chan-major (nticks, nchans)
    arrs = [a.T for a in arrs]
    extents = [(e[0][0], e[0][1], e[1][0], e[1][1]) for e in extents]

    for a,e in zip(arrs,extents):
        print a.shape, e


    fig = three_horiz(arrs, extents, name, baselines)
    fig.suptitle(tit)
    fig.savefig(out)


def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
    
