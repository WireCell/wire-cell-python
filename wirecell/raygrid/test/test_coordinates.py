import torch
from wirecell.raygrid.coordinates import Coordinates
from wirecell.raygrid.examples import symmetric_views


def do_coords_check(coords, view1, view2):
    r0 = coords.zero_crossings[view1, view2]
    r1 = coords.ray_crossing((view1,0), (view2,0))
    print(f'{r0=}')
    assert torch.all(r0 == r1) # 2 ways to get the same

    dr0 = coords.ray_crossing((view1,0), (view2,1)) - r0
    dr1 = coords.ray_jump[view1, view2]
    print(f'{dr0=}')
    assert torch.all(dr0 == dr1)

def test_coordinates():
    views = symmetric_views();
    print()
    for iview, view in enumerate(views):
        print(f'{iview=}')
        for iray, ray in enumerate(view):
            print(f'\t{iray=}')
            for ipt, pt in enumerate(ray):
                print(f'\t\t{ipt=},{pt}')

    coords = Coordinates(views)
    view1 = [2,3,4]
    view2 = [3,4,2]
    for v1, v2 in zip(view1, view2):
        do_coords_check(coords, v1, v2) # scalar
    do_coords_check(coords, view1, view2) # batched

    p0 = coords.pitch_location((2, 0), (3, 0), 4)
    print(f'{p0=}')

