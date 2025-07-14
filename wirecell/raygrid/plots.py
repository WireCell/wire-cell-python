import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
from wirecell.raygrid import tiling


def make_fig_2d(bb, title="2D Cartesian Space"):
    # Set plot limits and labels for clarity
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    (x0,x1), (y0,y1) = bb
        
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Y-coordinate')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig, ax

def save_fig(fig, fname):
    fig.savefig(fname)
    

def point_activity(ax, coords, points: torch.Tensor, activity_tensors: list[torch.Tensor]):
    """
    Visualizes randomly generated 2D points and their per-view activity tensors.

    Args:
        coords: An instance of the Coordinates class.
        points: A tensor of shape (nbatch, 2) representing the 2D Cartesian points.
        activity_tensors: A list of boolean tensors, where each tensor corresponds
                          to the activity for a specific view.

    Returns:
        A matplotlib.figure.Figure object.
    """

    # 1. Plot the randomly generated point locations as a 2D scatter plot
    ax.scatter(points[:, 0].cpu().numpy(), points[:, 1].cpu().numpy(),
               color='blue', alpha=0.6, label='Random Points', s=10)

    # Get a colormap for distinct colors for each view
    colors = plt.cm.get_cmap('viridis', len(activity_tensors))

    # 2. Draw each activity tensor as a "stepped" plot (represented by active strips)
    for view_idx, activity_tensor in enumerate(activity_tensors):
        p_dir = coords.pitch_dir[view_idx]
        c = coords.center[view_idx]
        p_mag = coords.pitch_mag[view_idx]

        # Calculate the direction orthogonal to pitch_dir (i.e., the ray direction).
        # This is the direction along which the 'histogram' width extends.
        # It's obtained by rotating the pitch_dir 90 degrees clockwise: (-py, px).
        ray_dir_vec = torch.tensor([-p_dir[1], p_dir[0]], dtype=torch.float32, device=p_dir.device)

        # Determine a suitable length for visualizing the active strips.
        # This length should visually span across the 100x100 domain.
        plot_length = 120.0 # Slightly larger than 100 to ensure it crosses the domain clearly
        half_plot_length = plot_length / 2.0

        # Find indices within the activity tensor where activity is True
        active_indices = torch.where(activity_tensor)[0]

        for idx in active_indices:
            # Calculate the central pitch value for this active strip (bin).
            # This is (index + 0.5) * pitch_magnitude along the pitch direction.
            pitch_center_value = (idx + 0.5) * p_mag

            # Calculate the 2D point on the view's central ray corresponding to this pitch value.
            # This is the "center" of the strip in 2D space.
            point_on_pitch_axis = c + pitch_center_value * p_dir

            # Define the start and end points of the line segment that represents the active strip.
            # This segment extends along the 'ray_dir_vec' (orthogonal to pitch_dir)
            start_point = point_on_pitch_axis - half_plot_length * ray_dir_vec
            end_point = point_on_pitch_axis + half_plot_length * ray_dir_vec

            # Plot the line segment.
            # Label only the first segment for each view to avoid duplicate legend entries.
            ax.plot([start_point[0].cpu().numpy(), end_point[0].cpu().numpy()],
                    [start_point[1].cpu().numpy(), end_point[1].cpu().numpy()],
                    color=colors(view_idx),  # Use a distinct color for each view
                    linestyle='-',
                    linewidth=3, # Make lines thicker for better visibility
                    alpha=0.3,
                    label=f'View {view_idx} Activity' if idx == active_indices[0] else None)

    # Add a legend to explain the plots.
    # Filter out duplicate labels for view activities to keep the legend clean.
    handles, labels = ax.get_legend_handles_labels()
    unique_labels_dict = {}
    for h, l in zip(handles, labels):
        unique_labels_dict[l] = h
    ax.legend(unique_labels_dict.values(), unique_labels_dict.keys(), loc='upper left', bbox_to_anchor=(1, 1))


def blobs_strips(coords, ax: plt.Axes, blobs: torch.Tensor):
    """
    Plots the blobs as filled translucent strips for each view.

    Args:
        ax: A matplotlib Axes object, preconfigured to span the desired 2D Cartesian space.
        coords: An instance of the Coordinates object.
        blobs: A tensor of shape (nblobs, nviews, 2) where `blobs[b, v, 0]` is the
               minimum pitch value and `blobs[b, v, 1]` is the maximum pitch value
               for blob 'b' in view 'v'. These define the half-open bounds [min_pitch, max_pitch).
    """
    nblobs = blobs.shape[0]
    nviews = blobs.shape[1]

    # Get a colormap to assign a unique color to each view's strips
    view_colors = plt.cm.get_cmap('tab10', nviews)

    # Define an extent for the strips along the ray direction to ensure they span the scene.
    # This value should be sufficiently large to cover the typical 100x100 plotting area.
    ray_span_extent = 200.0
    half_ray_span_extent = ray_span_extent / 2.0

    # 1. Draw each blob's representation as a filled strip for each view
    for blob_idx in range(nblobs):
        print(f'{blob_idx=}')
        for view_idx in range(nviews):
            print(f'{view_idx=}')
            lo_pitch_idx = blobs[blob_idx, view_idx, 0].item()
            hi_pitch_idx = blobs[blob_idx, view_idx, 1].item()

            # Skip if the pitch range is empty or invalid
            if hi_pitch_idx - lo_pitch_idx < 1:
                continue

            center = coords.center[view_idx].cpu().numpy()
            pitch_dir = coords.pitch_dir[view_idx].cpu().numpy()
            pitch_mag = coords.pitch_mag[view_idx].cpu().numpy()
            ray_dir = coords.ray_dir[view_idx].cpu().numpy()
            half_ray = ray_dir * half_ray_span_extent

            lo_pitch_vec = lo_pitch_idx * pitch_mag * pitch_dir
            hi_pitch_vec = hi_pitch_idx * pitch_mag * pitch_dir
            print(f'{lo_pitch_idx=} {lo_pitch_vec=} {pitch_mag=}')
            print(f'{hi_pitch_idx=} {hi_pitch_vec=} {pitch_mag=}')

            ul = center + half_ray + lo_pitch_vec
            ll = center - half_ray + lo_pitch_vec
            ur = center + half_ray + hi_pitch_vec
            lr = center - half_ray + hi_pitch_vec

            # Convert to NumPy array for matplotlib Polygon
            polygon_coords = np.array([ll,ul,ur,lr])

            # Create and add the translucent polygon patch to the axis
            patch = patches.Polygon(polygon_coords,
                                    closed=True,
                                    facecolor=view_colors(view_idx),
                                    alpha=0.1, # Translucent fill
                                    edgecolor=view_colors(view_idx),
                                    linewidth=0.5, # Thin edge for definition
                                    # Label only the first blob's strip for each view to avoid legend clutter
                                    label=f'View {view_idx} Strip' if blob_idx == 0 else None)
            ax.add_patch(patch)



def blobs_corners(coords,
                  ax: plt.Axes, 
                  blobs: torch.Tensor, crossings: torch.Tensor, insides: torch.Tensor):
    """
    Plots the blobs corners (crossing points inside)

    Args:
        ax: A matplotlib Axes object, preconfigured to span the desired 2D Cartesian space.
        coords: An instance of the Coordinates object.
        blobs: A tensor of shape (nblobs, nviews, 2) where `blobs[b, v, 0]` is the
               minimum pitch value and `blobs[b, v, 1]` is the maximum pitch value
               for blob 'b' in view 'v'. These define the half-open bounds [min_pitch, max_pitch).
        crossings: A tensor of shape (nblobs, nviews, nviews, 2) where
                   `crossings[b, v1, v2, :]` contains `(pitch_idx_v1, pitch_idx_v2)`
                   representing the pitch indices of the rays that cross for blob 'b'
                   between view v1 and v2.
        insides: A boolean tensor of shape (nblobs, nviews, nviews) where
                 `insides[b, v1, v2]` is True if the crossing between view v1 and v2
                 for blob 'b' is considered "inside" the blob.
    """
    nblobs = blobs.shape[0]
    nviews = blobs.shape[1]


    # Get a colormap to assign a unique color to each view's strips
    view_colors = plt.cm.get_cmap('tab10', nviews)

    # Define an extent for the strips along the ray direction to ensure they span the scene.
    # This value should be sufficiently large to cover the typical 100x100 plotting area.
    ray_span_extent = 150.0
    half_ray_span_extent = ray_span_extent / 2.0

    # 2. Plot "inside" crossing points
    inside_crossing_points_x = []
    inside_crossing_points_y = []

    # Define visual properties for the crossing points
    crossing_point_color = 'red'
    crossing_point_marker = 'x'
    crossing_point_size = 70 # Larger marker size for visibility
    crossing_point_alpha = 0.8

    # Iterate over all blobs and unique pairs of views (v1, v2)
    for blob_idx in range(nblobs):
        for pair_idx, view_pair_idx in enumerate(tiling.strip_pairs(nviews)):
            view1_idx, view2_idx = view_pair_idx

            for edges_idx in range(4):
                # Check if this specific crossing (for current blob and view pair) is "inside"
                if not insides[blob_idx, pair_idx, edges_idx].item():
                    continue

                # Extract the pitch indices for the crossing
                pitch1_idx = crossings[blob_idx, pair_idx, edges_idx, 0].item()
                pitch2_idx = crossings[blob_idx, pair_idx, edges_idx, 1].item()

                # Calculate the 2D Cartesian coordinates of the crossing point
                crossing_pt_2d = coords.ray_crossing((view1_idx, pitch1_idx), (view2_idx, pitch2_idx))

                # Add to lists for plotting if the point is valid (not NaN)
                if not torch.any(torch.isnan(crossing_pt_2d)):
                    inside_crossing_points_x.append(crossing_pt_2d[0].item())
                    inside_crossing_points_y.append(crossing_pt_2d[1].item())

    # Plot all collected "inside" crossing points
    if inside_crossing_points_x:
        ax.scatter(inside_crossing_points_x, inside_crossing_points_y,
                   color=crossing_point_color, marker=crossing_point_marker,
                   s=crossing_point_size, alpha=crossing_point_alpha,
                   label='Corners', zorder=3) # Higher zorder to ensure visibility

    # Finalize the legend. Collect unique handles and labels, as some labels might be None.
    handles, labels = ax.get_legend_handles_labels()
    unique_labels_dict = {}
    for h, l in zip(handles, labels):
        # Add only if label is not None (i.e., it's an actual label)
        if l is not None:
            unique_labels_dict[l] = h
    # Place the legend outside the plot area to avoid obscuring data
    ax.legend(unique_labels_dict.values(), unique_labels_dict.keys(),
              loc='upper left', bbox_to_anchor=(1, 1))

    # Note: Axis limits, title, labels, and aspect ratio are expected to be set by the caller
    # when configuring the `ax` object.

