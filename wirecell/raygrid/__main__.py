#!/usr/bin/env python
'''
The "wcpy raygrid" commands.
'''
import click
from wirecell.util.cli import log, context
from itertools import product

@context("raygrid")
def cli(ctx):
    """
    Wire-Cell Toolkit commands related to raygrid.
    """
    pass

@cli.command("plot-cells")

@click.option("-d", "--detector", default="uboone",
              help="Canonical detector name") 
@click.option("-a", "--anode", default=0,
              help="Anode (APA) index number") 
@click.option("-o", "--output", default=None,
              help="Filename for output graphics, else interactive") 
def cmd_plot_cells(detector, anode, output):
    """
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from wirecell.raygrid import wires as wmod
    from wirecell.raygrid.coordinates import Coordinates
    from wirecell.raygrid import tiling
    from wirecell.raygrid.plots import blobs_corners

    # wires array: (N, 2, 3) giving 2 3D endpoint from N wires.
    uwires, vwires, wwires = wmod.load(detector, anode)

    nu = len(uwires)
    nv = len(vwires)
    nw = len(wwires)
    nwires = (nu, nv, nw)

    sizes = (10, 10, 10)
    ranges = [list(range(n//2-s,n//2+s)) for n,s in zip(nwires, sizes)]

    uwires_sel = uwires[ranges[0]]
    vwires_sel = vwires[ranges[1]]
    wwires_sel = wwires[ranges[2]]

    all_selected = torch.vstack((uwires_sel, vwires_sel, wwires_sel))
    all_centers = 0.5*(all_selected[:, 0, :] + all_selected[:, 1, :])
    all_xy_centers = torch.vstack((all_centers[:, 2], all_centers[:, 1]))

    xy_center = torch.sum(all_xy_centers, axis = 1) / all_xy_centers.shape[1]
    print(f'{xy_center=}')


    def get_blobs():
        views = wmod.make_views(uwires, vwires, wwires, 100,100)
        coords = Coordinates(views)

        all_blobs = list()
        for winds in product(*ranges):
            blobs = tiling.trivial_blobs()
            for iplane, wind in enumerate(winds):
                n = nwires[iplane]
                activities = torch.zeros(n, dtype=torch.bool)
                activities[wind] = True
                new_blobs = tiling.apply_activity(coords, blobs, activities)
                if not new_blobs.shape[0]:
                    break
                blobs = new_blobs
            if blobs.shape[0]:
                print(blobs.shape)
                all_blobs.append(blobs)
                break
        return all_blobs
    #all_blobs = get_blobs()

    fig, ax = plt.subplots(1,1)
    ax.set_xlabel("Z [cm]")
    ax.set_ylabel("Y [cm]")
    ax.set_title(f"Wire Plane Projection (Detector: {detector}, Anode: {anode})")

    # Helper function to plot wires
    def plot_wires(wires, color, label):
        # wires shape is (N, 2, 3) -> (N wires, 2 endpoints, 3 coords (X, Y, Z))
        # We want Z (index 2) on 2D X axis, and Y (index 1) on 2D Y axis.

        if wires.size == 0:
            return

        # Extract Z coordinates: (N, 2)
        Z = wires[:, :, 2]
        # Extract Y coordinates: (N, 2)
        Y = wires[:, :, 1]

        N = wires.shape[0]

        # Stack Z and Y coordinates, inserting NaNs between segments for
        # continuous plotting This creates arrays of shape (3*N) where every 3rd
        # element is NaN, separating segments.
        nan_col = np.full((N, 1), np.nan)

        Z_plot = np.hstack([Z, nan_col]).reshape(-1)
        Y_plot = np.hstack([Y, nan_col]).reshape(-1)
        ax.plot(Z_plot, Y_plot, color=color, label=label, linewidth=0.5,
                linestyle='solid')

    plot_wires(uwires[ranges[0]], 'red', 'U Wires')
    plot_wires(vwires[ranges[1]], 'blue', 'V Wires')
    plot_wires(wwires[ranges[2]], 'green', 'W Wires')

    x_cen, y_cen = xy_center
    psize = 20
    ax.set_xlim(x_cen-psize, x_cen+psize)
    ax.set_ylim(y_cen-psize, y_cen+psize)

    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    if output:
        plt.savefig(output)
    else:
        plt.show()

def main():
    cli(obj=dict())

if '__main__' == __name__:
    main()
