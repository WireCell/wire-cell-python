#!/usr/bin/env python

from ROOT import *

nwires = 0
lines = []
# with open('protodunevd-wires-larsoft-v1.txt') as file:
with open('protodunevd-wires-larsoft-v1.txt-110120.txt-splitanode.txt') as file:
  for txtline in file:
    if txtline.startswith('#'): continue    
    content = txtline.strip('\n').split()
    channel = int(content[0])
    tpc     = int(content[1])
    plane   = int(content[2])
    wire    = int(content[3])
    sx      = float(content[4])*0.01 #m 
    sy      = float(content[5])*0.01 #m
    sz      = float(content[6])*0.01 #m
    ex      = float(content[7])*0.01 #m
    ey      = float(content[8])*0.01 #m
    ez      = float(content[9])*0.01 #m

    if tpc == 110 and plane == 1:
      nwires += 1
      aline = TLine(sz,sy, ez, ey)
      # if channel<1145 or channel>1240: continue
      if channel==1145: lines.append(aline)
      print(txtline.strip('\n'))

print(f'nwires= {nwires}, lines.size: {len(lines)}')

c1 = TCanvas("c1","",400,800)
c1.cd()
h2 = TH2F("h2","",10,-3,3,10,-3.5,3.5)
h2.GetXaxis().SetTitle("Z (m)")
h2.GetYaxis().SetTitle("Y (m)")
h2.Fill(1.5,1.5)
h2.Draw("")

for line in lines[1:]:
  # print(f"{line.GetX1()} {line.GetY1()} {line.GetX2()} {line.GetY2()}")
  line.Draw("same")
