import torch
import matplotlib.pyplot as plt
from wirecell.raygrid.examples import Rasterizer3D
from wirecell.raygrid.examples import symmetric_views
from wirecell.raygrid.coordinates import Coordinates
from wirecell.raygrid import tiling

def test_line_tiling():

    grid_center = torch.tensor([5.0, 0.0, 0.0], dtype=torch.float32) # Centered on the line
    grid_size = (10.0, 10.0)
    grid_pixels = (100, 100)

    coords = Coordinates(symmetric_views())

    rasters = list()
    for pdir in coords.pitch_dir[2:]: # 2D
        angle = pdir[1].item()
        print(f'{angle=} {180.0 * angle/torch.pi}deg')
        raster = Rasterizer3D(
            grid_center,
            grid_size[0], grid_size[1],
            grid_pixels[0], grid_pixels[1],
            angle)
        track = torch.tensor([[1.0, 0.0, 0.0], [9.0, 3.0, 1.0]], dtype=torch.float32)
        rasters.append(raster)
    
    for letter, raster in zip("uvw", rasters):
        arr = raster.get_raster().detach().cpu().numpy()
        plt.title(f'{letter} view of a simple rastered track')
        plt.imshow(arr, cmap='viridis', origin='upper')
        plt.savefig(f'test-line-tiling-view-{letter}.png')
        plt.clf();

    
    for tick in range(100):
        img = tiling.trivial()
        for letter, raster in zip("uvw", rasters):
            view = raster.get_raster()
            activity = view[:,tick] > 0
            if not torch.any(activity):
                print(f'no activity in view {letter} tick {tick}')
                continue
            img = tiling.apply_view(coords, img, activity)
            if img is None:
                print(f'tick {tick} has no blobs')
                break
            print(f'tick {tick} has {img.nblobs} blobs')
            
