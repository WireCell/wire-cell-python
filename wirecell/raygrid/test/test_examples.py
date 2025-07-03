import torch
import matplotlib.pyplot as plt
from wirecell.raygrid.examples import symmetric_views

def test_symmetric_views():
    colors = ['black','black','red','green','blue']

    views = symmetric_views();
    
    
    for iview, view in enumerate(views):
        print(f'{iview=}')
        for iray, ray in enumerate(view):
            print(f'\t{iray=}')
            plt.plot(ray[:,0], ray[:,1], color=colors[iview])
            plt.text(0.5*torch.sum(ray[:,0]), 0.5*torch.sum(ray[:,1]), f'{iray}')
            for ipt, pt in enumerate(ray):
                print(f'\t\t{ipt=},{pt}')
    outfile = "test_symmetric_views.pdf"
    print(outfile)
    plt.savefig(outfile)
