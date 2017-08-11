import ROOT


def open_file(fileuri, mode = "READONLY"):
    tfile = ROOT.TFile.Open(fileuri, mode)
    if not tfile:
        raise IOError('failed to open %s' % fileuri)
    return tfile
    

def load_obj(tfile, name):
    obj = tfile.Get(name)
    if not obj:
        raise IOError('failed to get hist "%s" from %s' % (name, fileuri))
    obj.SetDirectory(0)
    return obj


def save(obj, fname):
    tfile = ROOT.TFile.Open(fname, "recreate")
    if not tfile:
        raise IOError('failed to open %s' % fname)
    obj.SetDirectory(tfile)
    tfile.Write()
    tfile.Close();
    

def is_hist(obj):
    if obj.InheritsFrom("TKey"):
        return obj.GetClassName()[:3] in ["TH1", "TH2", "TH3"]
    return obj.InheritsFrom("TH1")

def hist_to_dict(hist):
    '''
    Return data structure summarizing the histogram.
    '''
    ndim = int(hist.ClassName()[2])
    dims = list()
    for idim in range(ndim):
        letter = "XYZ"[idim]
        nbins = getattr(hist, "GetNbins"+letter)()
        axis = getattr(hist, "Get"+letter+"axis")()
        lo = axis.GetBinLowEdge(1);
        hi = axis.GetBinUpEdge(nbins);
        dim = dict(letter=letter, nbins=nbins, min=lo, max=hi, rms=hist.GetRMS(idim+1))
        dims.append(dim)

    return dict(axis=dims, name=hist.GetName(), title=hist.GetTitle(),
                integ=hist.Integral(),
                min=hist.GetMinimum(), max=hist.GetMaximum())

    
def resize_hist2f(hist, xrebin, yrebin):
    '''
    Rebin 2D histogram.  This is dog slow.
    '''
    xaxis = hist.GetXaxis()
    nbinsx = hist.GetNbinsX()
    yaxis = hist.GetYaxis()
    nbinsy = hist.GetNbinsY()

    h = ROOT.TH2F(hist.GetName(), hist.GetTitle(),
                  int(round(nbinsx/xrebin)), xaxis.GetBinLowEdge(1), xaxis.GetBinUpEdge(nbinsx),
                  int(round(nbinsy/yrebin)), yaxis.GetBinLowEdge(1), yaxis.GetBinUpEdge(nbinsy))

    xaxis_new = h.GetXaxis()
    yaxis_new = h.GetYaxis()

    for ixbin in range(hist.GetNbinsX()):
        x = xaxis.GetBinCenter(ixbin+1)
        ix = xaxis_new.FindBin(x)
        print x,ix
        for iybin in range(hist.GetNbinsY()):
            y = yaxis.GetBinCenter(iybin+1)
            iy = yaxis_new.FindBin(y)
            val = hist.GetBinContent(ixbin+1, iybin+1)
            ibin = h.GetBin(ix,iy)
            h.AddBinContent(ibin, val)
    return h

