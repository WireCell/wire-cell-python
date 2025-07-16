import torch
from wirecell.raygrid.coordinates import Coordinates
from wirecell.raygrid.examples import symmetric_views


def do_coords_check(coords, view1, view2):
    r0 = coords.zero_crossings[view1, view2]
    r1 = coords.ray_crossing(view1,0, view2,0)
    print(f'{r0=}')
    assert torch.all(r0 == r1) # 2 ways to get the same

    dr0 = coords.ray_crossing(view1,0, view2,1) - r0
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
    print(coords.as_string("c++"))

    view1 = [2,3,4]
    view2 = [3,4,2]
    for v1, v2 in zip(view1, view2):
        do_coords_check(coords, v1, v2) # scalar
    do_coords_check(coords, view1, view2) # batched

    p0 = coords.pitch_location(2, 0, 3, 0, 4)
    print(f'{p0=}')

def test_three():
    # translation of C++ test
    views = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0]], # View 0: horizontal, pitch along x-axis
        [[0.0, 0.0], [0.0, 1.0]], # View 1: vertical, pitch along y-axis
        [[0.0, 0.0], [0.70710678, 0.70710678]]]) # View 2: diagonal, pitch along y=x
    coords = Coordinates(views)
    
    # (nbatch, 2): [(view, ray), ...]
    one_batch = torch.tensor([
        [0, 0], [0, 1], [0,1] 
    ], dtype=torch.long)
    two_batch = torch.tensor([
        [1, 0], [1, 0], [1,1]
    ], dtype=torch.long)
    print(f'{one_batch.shape=}')
    crossing_point_batch = coords.ray_crossing(one_batch[:,0], one_batch[:,1], two_batch[:,0], two_batch[:,1]);
    print(f"ray_crossing (batch):\n{crossing_point_batch}")
    # (view 0, ray 0) and (view 1, ray 0) -> (0,0)
    # (view 0, ray 1) and (view 1, ray 0) -> (1,0) (ray 1 of view 0 is x=1, ray 0 of view 1 is y=0)
    expected_crossing_batch = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ], dtype=torch.float);
    assert torch.all(crossing_point_batch == expected_crossing_batch)

    three_batch = torch.tensor([2, 2, 2])
    pl = coords.pitch_location(one_batch[:,0], one_batch[:,1], two_batch[:,0], two_batch[:,1], three_batch);
    print(f'{pl=}')
    expected_pl = torch.tensor([0, 0.70710678, 2*0.70710678])
    assert torch.allclose(pl, expected_pl)

def test_dump_coordinates():
    views = symmetric_views();
    coords = Coordinates(views)
    torch.set_printoptions(precision=17)
    with open("test-dump-coordinates.cpp","w") as fp:
        fp.write(coords.as_string("c++"))
        fp.write("\n")
    print('Dumped C++ snippet holding Coordinate data to "test-dump-coordinates.cpp"')
    
