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
ROOT.gStyle.SetNumberContours(100)

def set_palette(which = "custom"):
    if not which or which == "custom":
        ROOT.TColor.CreateGradientColorTable(len(stops), stops, reds, greens, blues, 100)
        return
    ROOT.gStyle.SetPalette(which)
 

fp = ROOT.TFile.Open("build/util/test_impactresponse.root")
assert (fp)
fp.ls()

outname = "test_impactresponse.pdf"

c = ROOT.TCanvas()
c.SetRightMargin(0.15)
c.Print(outname+"[","pdf")
c.SetGridx()
#c.SetGridy()

limits = dict(fr = [0.5e-12, 0.3e-12, 0.8e-12],
              dr = [9.0e-12, 9.0e-12, 20.0e-12])
lmins = dict(fr = [-17, -17, -17],
              dr = [-15, -15, -15])


wline = ROOT.TLine()                               # wire line
wline.SetLineColorAlpha(1, 0.5)
wline.SetLineStyle(1)

hline = ROOT.TLine()                               # half line
hline.SetLineColorAlpha(2, 0.5)
hline.SetLineStyle(2)
def draw_wires():
    for wpitch in range(0,36,3):
        wline.DrawLine(0,  wpitch, 100,  wpitch)
        wline.DrawLine(0, -wpitch, 100, -wpitch)
        if wpitch:
            hline.DrawLine(0,  wpitch-1.5, 100,  wpitch-1.5)
            hline.DrawLine(0, -wpitch+1.5, 100, -wpitch+1.5)

for iplane, letter in enumerate("UVW"):
    for name in ["fr","dr"]:
        hist = fp.Get("h%s_%c" % (name, letter))
        assert(hist)

        set_palette()
        hist.Draw("colz")
        draw_wires()
        lim = limits[name][iplane]
        hist.GetZaxis().SetRangeUser(-lim, lim)
        c.Print(outname,"pdf")

        set_palette(ROOT.kRainBow)
        lminin = lmins[name][iplane]
        lminout = bilogify(hist, lminin)
        print name,iplane,lminin,lminout
        title = hist.GetTitle()
        title += " [sign(z)(log10(abs(z)) %d)]" % lminout
        hist.SetTitle(title)
        hist.Draw("colz")
        draw_wires()
        c.Print(outname,"pdf")

c.Print(outname+"]","pdf")
