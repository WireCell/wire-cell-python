import torch
from wirecell.raygrid.coordinates import Coordinates
from wirecell.raygrid.examples import symmetric_views


def test_construct():
    views = symmetric_views();
    print()
    for iview, view in enumerate(views):
        for iray, ray in enumerate(view):
            for ipt, pt in enumerate(ray):
                print(f'{iview},{iray},{ipt},{pt}')

    coords = Coordinates(views)
    
