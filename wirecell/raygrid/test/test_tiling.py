import math
import torch
import matplotlib.pyplot as plt
from wirecell.raygrid.examples import Raster
from wirecell.raygrid.examples import symmetric_views
from wirecell.raygrid.coordinates import Coordinates
from wirecell.raygrid import tiling

def print_cb(coords, blobs):
    for iblob, blob in enumerate(blobs):
        for iview, view in enumerate(blob):
            print(f'blob {iblob} view {iview}: {view}, pitch={coords.pitch_mag[iview] * coords.pitch_dir[iview]}')


def test_individual_steps():

    coords = Coordinates(symmetric_views())

    #
    # From 2 to 3 layers
    # 
    b1 = torch.tensor([ [ [0,1], [0,1] ] ], dtype=torch.long )
    print(f'{b1=}')
    c1 = tiling.blob_crossings(b1)
    print(f'{c1=}')
    i1 = tiling.blob_insides(coords, b1, c1)
    assert torch.all( i1 )
    print(f'{i1=}')
    lo, hi = tiling.blob_bounds(coords, c1, 2, i1)
    assert(lo[0] == -1)
    assert(hi[0] == 46)

    nmeasures2=40               # size of view 2 (third view)
    lo2, hi2 = tiling.bounds_clamp(lo, hi, nmeasures2)
    print(f'{lo2=}, {hi2=}')
    assert(lo2[0] == 0)
    assert(hi2[0] == 40)
    
    act2 = torch.zeros(nmeasures2, dtype=torch.bool)
    act2[:10] = True            # edge
    act2[20:30] = True          # middle
    act2[35:] = True            # edge
    b2 = tiling.expand_blobs_with_activity(b1, lo2, hi2, act2)
    print(f'{b2=}')

    b2prime = tiling.apply_activity(coords, b1, act2)
    assert torch.all(b2 == b2prime)


def test_simple_tiling():

    coords = Coordinates(symmetric_views())
    img1 = tiling.trivial()
    assert img1.blobs.shape[0] == 1 # blobs
    assert img1.blobs.shape[1] == 2 # views
    assert torch.all(img1.crossings < 2) # special layer, no ray indices larger than 1


    # ## interlude to test projection
    # print_cb(coords, img1.blobs)
    # print("IMG1:", img1.as_string(depth=3))
    # proj1 = tiling.projection(coords, img1)
    # print(f'{proj1=}')
    # assert proj1[0].item() == -1 # tofu
    # assert proj1[1].item() == 45 # tofu

    # activity2 = torch.tensor([0,0,0,1,1,1,0,0,1,1,0,1,1])>0.5
    # img2 = tiling.apply_view(coords, img1, activity2)
    # print_cb(coords, img2.blobs)    

def test_fresh_crossings():
    fresh_blobs = torch.tensor([[
        [ 0,  1],               # blob 0, strip 0
        [ 0,  1],               # blob 0, strip 1
        [ 3,  6]                # blob 0, strip 2
    ], [
        [ 0,  1],               # blob 1
        [ 0,  1],
        [ 8, 10]
    ], [
        [ 0,  1],               # blob 2
        [ 0,  1],
        [11, 13]]], dtype=torch.long)

    old_crossings = torch.tensor([[[
        [0, 0],
        [1, 0]
    ], [
        [0, 0],
        [1, 1]
    ], [
        [0, 1],
        [1, 0]
    ], [
        [0, 1],
        [1, 1]]]], dtype=torch.long)
    fresh_crossings = tiling.fresh_crossings(fresh_blobs, old_crossings)

def test_fresh_insides():
    from sampledata import all_blobs, all_crossings
    fi = tiling.fresh_insides(all_blobs, all_crossings)


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


        grid_normal = torch.tensor([pdir[0], pdir[1], 0.0]) # pitch

        raster = Raster(grid_normal, grid_center, grid_direction, grid_size, grid_shape)

        # 
        theta_x_deg = math.atan2(pdir[1].item(), pdir[0].item()) * 180.0 / math.pi
        theta_y_deg = 90 - theta_x_deg
        print(f'{cind=} {grid_size=} {grid_shape=} {theta_x_deg=} {theta_y_deg=}')
        print(f'\t{grid_normal=}')
        print(f'\t{grid_center=}')
        print(f'\t{grid_direction=}')
        raster.add_line(torch.tensor([0.0, 0.0, -8.0]), torch.tensor([0.0, 0.0, 8.0]))
        raster.add_line(torch.tensor([25.0, 25.0, -8.0]), torch.tensor([-25.0, -25.0, 8.0]))
        rasters.append(raster)
    
    pix_per=2
    rows, cols = grid_shape
    fig_width_pixels = cols * pix_per
    fig_height_pixels = rows * pix_per
    dpi = 100
    fig_width_inches = fig_width_pixels / dpi
    fig_height_inches = fig_height_pixels / dpi

    for letter, raster in zip("uvw", rasters):
        arr = raster.pixels.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches), dpi=dpi)
        plt.title(f'{letter} view of a simple rastered track')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.imshow(arr, cmap='viridis', origin='upper', interpolation='none')
        plt.savefig(f'test-line-tiling-view-{letter}.png',
                    dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.clf()

    
    ntick_empty=0
    nblobs_empty = 0

    for tick in range(grid_shape[0]):
        img = tiling.trivial()
        print(f'trivial: {img.blobs=}')


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
            print(f'view {letter}: {img.blobs=}')
            
            print(f'tick {tick} has {img.nblobs} blobs')
        if nview_empty:
            ntick_empty += 1

    if ntick_empty:
        print(f'no activity in {ntick_empty} ticks')
    if nblobs_empty:
        print(f'no blobs in {nblobs_empty} ticks')
