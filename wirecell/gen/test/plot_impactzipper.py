#!/usr/bin/env python

from math import log10
import ROOT

def bilogify(hist, lmin = None):
    """
    Modify histogram so that it is rescaled in a "bilog" manner.
    Negative values are -log10(-z).  An offset in log10 is returned
    and represents the 0.
    """
    nx,ny = hist.GetNbinsX(), hist.GetNbinsY()
    zmax = max(abs(hist.GetMinimum()), abs(hist.GetMaximum()))

    if lmin is None:
        zmin = sum([abs(hist.GetBinContent(99,iy)) for iy in range(ny)]) / ny
        lmin = int(log10(zmin))

    lmax = 1+int(log10(zmax))

    for ix in range(hist.GetNbinsX()):
        for iy in range(hist.GetNbinsY()):
            val = hist.GetBinContent(ix, iy)
            if val == 0.0:
                hist.SetBinContent(ix, iy, 0.0)
                continue
            
            sign = 1.0
            if val > 0:                   # opposite sign to match Xin 
                sign = -1.0
            lval = log10(abs(val))
            if lval < lmin:
                lval = 0.0
            else:
                lval -= lmin
                lval *= sign
            hist.SetBinContent(ix, iy, lval)
            continue
        continue

    lhmax = lmax - lmin

    hist.SetMaximum(lhmax)
    hist.SetMinimum(-lhmax)
    return lmin


from array import array
stops = array('d',[ 0.00, 0.45, 0.50, 0.55, 1.00 ])
reds =  array('d',[ 0.00, 0.00, 1.00, 1.00, 0.51 ])
greens =array('d',[ 0.00, 0.81, 1.00, 0.20, 0.00 ])
blues  =array('d',[ 0.51, 1.00, 1.00, 0.00, 0.00 ])

#ROOT.gStyle.SetPalette(ROOT.kVisibleSpectrum)
#ROOT.TColor.CreateGradientColorTable(len(stops), stops, reds, greens, blues, 100)
ROOT.gStyle.SetNumberContours(100)

def set_palette(which = "custom"):
    if not which or which == "custom":
        ROOT.TColor.CreateGradientColorTable(len(stops), stops, reds, greens, blues, 100)
        return
    ROOT.gStyle.SetPalette(which)

fp = ROOT.TFile.Open("build/gen/test_impactzipper-uvw.root")
hists = {u:fp.Get("h%d"%n) for n,u in enumerate("uvw")}
limits = [1,1,2]
lmins = [-3, -3, -3]

pdffile="plot_impactzipper.pdf"

c = ROOT.TCanvas()
c.SetRightMargin(0.15)
c.Print(pdffile+"[","pdf")
c.SetGridx()
c.SetGridy()

for (p,h),lim,lmin in zip(sorted(hists.items()), limits, lmins):
    print p

    set_palette()
    h.Draw("colz")
    h.SetTitle(h.GetTitle() + " point source")
    h.GetXaxis().SetRangeUser(3900, 4000)
    h.GetYaxis().SetRangeUser(989, 1012)
    h.GetZaxis().SetRangeUser(-lim, lim)
    c.Print(pdffile,"pdf")

    set_palette(ROOT.kRainBow)
    lminout = bilogify(h, lmin)
    title = h.GetTitle()
    title += " [sign(z)(log10(abs(z)) %d)]" % lminout
    h.SetTitle(title)
    h.SetZTitle("")
    h.Draw("colz")
    c.Print(pdffile,"pdf")


c.Print(pdffile+"]","pdf")
