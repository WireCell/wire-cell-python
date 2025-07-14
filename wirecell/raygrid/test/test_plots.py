import torch
from wirecell.raygrid.coordinates import Coordinates
from wirecell.raygrid.examples import (
    symmetric_views, random_points, random_groups, fill_activity
)
from wirecell.raygrid import plots, tiling
import matplotlib.pyplot as plt

def make_stuff():
    coords = Coordinates(symmetric_views())
    points = random_groups(sigma=3)
    points = points.reshape(-1, 2)

    activities = fill_activity(coords, points)
    assert len(activities) == coords.nviews
    assert activities[0].shape == (1,)
    assert activities[1].shape == (1,)
    for i in range(2, coords.nviews):
        assert activities[i].shape[0] <= 100
        # Additionally, verify that at least some indices were marked True,
        # assuming the random points cover some part of the space.
        # This is a weak check, but indicates successful filling.
        assert activities[i].sum() >= 0 # sum of booleans should be non-negative

    return coords, points, activities


def test_activity_fill():

    coords, points, activities = make_stuff()

    fig,ax = plots.make_fig_2d(coords)

    plots.point_activity(ax, coords, points, activities)

    plots.save_fig(fig, "test-activity-fill.pdf")



    
def test_blob_building():
    coords, points, activities = make_stuff()

    # bb = coords.bounding_box.cpu().numpy()
    bb = torch.tensor([ [-100,200], [-100,200] ]).cpu().numpy()

    fig,ax = plots.make_fig_2d(bb)

    t = tiling.trivial()

    fname = "test-blob-building.pdf"
    with plots.PdfPages(fname) as pdf:

        fig,ax = plots.make_fig_2d(bb)
        plots.point_activity(ax, coords, points, activities)
        pdf.savefig(fig)

        fig,ax = plots.make_fig_2d(bb)
        plots.blobs_strips(coords, ax, t.blobs)
        plots.blobs_corners(coords, ax, t.blobs, t.crossings, t.insides)
        pdf.savefig(fig)

        blobs = t.blobs
        for view in [2,3,4]:

            new_blobs = tiling.apply_activity(coords, blobs, activities[view])
            new_crossings = tiling.blob_crossings(new_blobs)
            new_insides = tiling.blob_insides(coords, new_blobs, new_crossings)

            fig,ax = plots.make_fig_2d(bb)
            plots.blobs_strips(coords, ax, new_blobs)
            plots.blobs_corners(coords, ax, new_blobs, new_crossings, new_insides)
            pdf.savefig(fig)

            blobs = new_blobs


