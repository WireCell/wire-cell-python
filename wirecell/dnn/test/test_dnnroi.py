import pytest
from wirecell.dnn.data.common import DatasetParams
from wirecell.dnn.apps import dnnroi
from glob import glob
from itertools import product
@pytest.fixture
def recps():
    ret = glob("/nfs/data/1/bviren/dnnroi/data/g4-rec-*.h5")
    assert ret
    return ret
@pytest.fixture
def trups():
    ret = glob("/nfs/data/1/bviren/dnnroi/data/g4-tru-*.h5")
    assert ret
    return ret

def test_data(recps, trups):
    drec = dnnroi.data.Rec(recps)
    dtru = dnnroi.data.Tru(trups)
    print(f'{drec[0].shape=} {dtru[0].shape=}')

# pytest --durations=0 -s -k 'test_data_timing' test/test_dnnroi.py
#
# About 2-2.25 seconds to load one rec file.
# With no grad, about 300 MB GB RES per file.
# With grad about 2 GB RES per file.
# A few % variance in time run-to-run.
# No clear benefit/drawback to lazy vs eager loading.
# Clear RAM/time tradeoff with/without cache when multiple iterations expected.
#
# Disable test by default as it is slow and it does not actually assert anything.
#@pytest.mark.skip(reason="slow, see comments")
@pytest.mark.parametrize("lazy,cache,grad,nloads,nfiles",
                         product([True],[True],[True],[2],[2]))  # hack to test variants
def test_data_timing(recps, lazy, cache, grad, nloads, nfiles):
    dsparams = DatasetParams(lazy, cache, grad)
    drec = dnnroi.data.Rec(recps[:nfiles], dsparams=dsparams)
    while nloads:
        nloads -= 1
        for idx in range(len(drec)):
            drec[idx]

