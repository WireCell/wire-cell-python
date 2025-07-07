import torch
import matplotlib.pyplot as plt
from wirecell.raygrid.examples import symmetric_views

def test_symmetric_views():
    colors = ['black','black','red','green','blue']

    pitch_rays = symmetric_views();
    print(f'{pitch_rays.shape=}')
    
    
    plt.figure(figsize=(10,10))
    for iview, pray in enumerate(pitch_rays):
        print(f'{iview=} {pray.shape=}')
        beg,end = pray
        pvec = end - beg
        pdir = pvec / torch.linalg.norm(pvec)

        wdir = 10*torch.tensor([-pdir[1], pdir[0]])

        plt.arrow(beg[0], beg[1], pvec[0], pvec[1], head_width=1.0, head_length=1.0)
        plt.arrow(beg[0], beg[1], wdir[0], wdir[1], head_width=1.0, head_length=1.0, color='red')
        plt.text(beg[0]+pvec[0], beg[1]+pvec[1], f'P{iview}')
        plt.text(beg[0]+wdir[0], beg[1]+wdir[1], f'W{iview}', color='red')

        # for iray, ray in enumerate(pray):
        #     print(f'\t{iray=} {ray.shape=}')
        #     plt.plot(ray[:,0], ray[:,1], color=colors[iview])
        #     plt.text(0.5*torch.sum(ray[:,0]), 0.5*torch.sum(ray[:,1]), f'{iray}')
        #     for ipt, pt in enumerate(ray):
        #         print(f'\t\t{ipt=},{pt}')

    plt.text(100, 60, f'X_rg')
    plt.text(60, 100, f'Y_rg')
    plt.text(52,52, "(Z_rg) (out of page)")

    outfile = "test_symmetric_views.pdf"
    print(outfile)
    plt.savefig(outfile)
