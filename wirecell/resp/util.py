from wirecell.sigproc.response.arrays import pr2array
from wirecell.util import lmn

def pr2sigs(pr, period):
    '''
    Return list of signals for path responses in plane response pr.
    '''
    return [
        lmn.Signal(lmn.Sampling(T=period, N=imp.current.size),
                   wave=imp.current, name=str(imp.pitchpos))
        for imp in pr.paths
    ]


def fr2sigs(fr):
    '''
    Return list of list of response signals for each plane.
    '''
    return [pr2sigs(pr, fr.period) for pr in fr.planes]

