import math
import torch
import matplotlib.pyplot as plt
from wirecell.raygrid.examples import Raster
from wirecell.raygrid.examples import symmetric_views
from wirecell.raygrid.coordinates import Coordinates
from wirecell.raygrid import tiling

def test_point_tiling():

    coords = Coordinates(symmetric_views())
    

def test_line_tiling():

    grid_size = (10.0, 10.0)
    grid_shape = (100, 100)
    grid_center = torch.tensor([0.0,0.0,0.0])
    grid_direction = torch.tensor([0.0,0.0,1.0])

    coords = Coordinates(symmetric_views())

    # Here we will work in the coordinate system of the rays.  This puts z_rg
    # into the page.  This is NOT the WCT coordinate system.

    rasters = list()
    for cind in [2,3,4]:
        pdir = coords.pitch_dir[cind]
        x_i, y_i = coords.center[cind]

        cosine_y = pdir[0].item()
        angle = math.acos(cosine_y)

        grid_center = torch.tensor([x_i, y_i, 0.0])
        grid_normal = torch.tensor([pdir[0], pdir[1], 0.0])

        raster = Raster(grid_normal, grid_center, grid_direction, grid_size, grid_shape)

        raster.add_line(torch.tensor([1.0, 0.0, -8.0]), torch.tensor([5.0, 3.0, 8.0]))
        rasters.append(raster)
    
    for letter, raster in zip("uvw", rasters):
        arr = raster.pixels.detach().cpu().numpy()
        plt.title(f'{letter} view of a simple rastered track')
        plt.imshow(arr, cmap='viridis', origin='upper')
        plt.savefig(f'test-line-tiling-view-{letter}.png')
        plt.clf();

    
    ntick_empty=0
    nblobs_empty = 0

    for tick in range(100):
        img = tiling.trivial()

        nview_empty = 0
        for letter, raster in zip("uvw", rasters):
            view = raster.pixels
            activity = view[:,tick] > 0
            if not torch.any(activity):
                nview_empty += 1
                continue

            img = tiling.apply_view(coords, img, activity)
            if img is None:
                nblobs_empty += 1
                break
            
            print(f'tick {tick} has {img.nblobs} blobs')
        if nview_empty:
            ntick_empty += 1

    if ntick_empty:
        print(f'no activity in {ntick_empty} ticks')
    if nblobs_empty:
        print(f'no blobs in {nblobs_empty} ticks')
