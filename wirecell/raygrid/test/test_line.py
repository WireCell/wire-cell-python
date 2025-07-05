import torch
from wirecell.raygrid.examples import rasterize_3d_line_segment_to_2d_grid

def test_rasterize_3d_line_segment_to_2d_grid():
    # --- Example Usage 1: Simple case, line parallel to X-axis, grid is XZ plane ---
    print("--- Example 1: Line parallel to X-axis, grid is XZ plane ---")
    line_segment_ex1 = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=torch.float32)
    grid_center_ex1 = torch.tensor([2.5, 0.0, 0.0], dtype=torch.float32) # Centered on the line
    grid_total_width_ex1 = 10.0
    grid_total_height_ex1 = 10.0
    num_pixels_width_ex1 = 5
    num_pixels_height_ex1 = 5
    grid_angle_y_radians_ex1 = 0.0 # Grid is in XZ plane (normal along Y)

    raster_ex1 = rasterize_3d_line_segment_to_2d_grid(
        line_segment_ex1, grid_center_ex1, grid_total_width_ex1,
        grid_total_height_ex1, num_pixels_width_ex1, num_pixels_height_ex1,
        grid_angle_y_radians_ex1
    )
    print("Rasterized Grid (Example 1):\n", raster_ex1)
    # Expected: The middle row should have values, total length 5.0 spread over 5 pixels means 1.0 per pixel.
    # Assuming (0,0) is bottom-left, the projected line is at v=0.
    # pixel height is 2.0, total height 10.0. range is -5 to 5.
    # If line is at v=0, and center is at 0, it falls into the middle row (index 2).
    # If num_pixels_height = 5, then pixel_v_idx = 2 maps to row num_pixels_height - 1 - 2 = 5-1-2 = 2
    # So row 2 should have values.
    # The line segment (0,0,0) to (5,0,0) is 5 units long.
    # grid width is 10, num_pixels_width 5. pixel width is 2.
    # So line goes from u = -2.5 to u = 2.5.
    # Pixel boundaries:
    # u_0: [-5, -3]
    # u_1: [-3, -1]
    # u_2: [-1, 1]  (contains u=0)
    # u_3: [1, 3]   (contains u=2.5)
    # u_4: [3, 5]
    # The line (0 to 5) projects from u=-2.5 to u=2.5.
    # Pixels:
    # u_0 covers [-5, -3] -> no line
    # u_1 covers [-3, -1] -> 0.5 unit (from -2.5 to -1)
    # u_2 covers [-1, 1]  -> 2.0 units (from -1 to 1)
    # u_3 covers [1, 3]   -> 2.0 units (from 1 to 2.5)
    # u_4 covers [3, 5] -> no line
    # Total length: 0.5 + 2.0 + 2.0 = 4.5. This is wrong.
    # The line segment is (0,0,0) to (5,0,0). Grid center is (2.5,0,0).
    # line_start_2d_u = torch.dot( (0,0,0)-(2.5,0,0), (1,0,0)) = -2.5
    # line_end_2d_u = torch.dot( (5,0,0)-(2.5,0,0), (1,0,0)) = 2.5
    # So projected line is from u=-2.5 to u=2.5. Total projected length 5.0.
    # pixel_width_3d = 10.0 / 5 = 2.0
    # grid_min_u_coord = -5.0
    #
    # Pixel u ranges:
    # pu_idx=0: [-5.0, -3.0]
    # pu_idx=1: [-3.0, -1.0]
    # pu_idx=2: [-1.0, 1.0]
    # pu_idx=3: [1.0, 3.0]
    # pu_idx=4: [3.0, 5.0]
    #
    # Intersection with [-2.5, 2.5]:
    # pu_idx=0: No
    # pu_idx=1: Intersects [-2.5, -1.0], length = 1.5
    # pu_idx=2: Intersects [-1.0, 1.0], length = 2.0
    # pu_idx=3: Intersects [1.0, 2.5], length = 1.5
    # pu_idx=4: No
    # Total length: 1.5 + 2.0 + 1.5 = 5.0. This seems correct.
    # Expected output for row 2 (index 2): [0, 1.5, 2.0, 1.5, 0]


    # --- Example Usage 2: Line at an angle, rotated grid ---
    print("\n--- Example 2: Line at an angle, rotated grid ---")
    line_segment_ex2 = torch.tensor([[0.0, 0.0, 0.0], [3.0, 3.0, 0.0]], dtype=torch.float32)
    grid_center_ex2 = torch.tensor([1.5, 1.5, 0.0], dtype=torch.float32)
    grid_total_width_ex2 = 4.0
    grid_total_height_ex2 = 4.0
    num_pixels_width_ex2 = 4
    num_pixels_height_ex2 = 4
    grid_angle_y_radians_ex2 = torch.pi / 4 # 45 degrees, grid normal is (0, cos(45), sin(45))

    raster_ex2 = rasterize_3d_line_segment_to_2d_grid(
        line_segment_ex2, grid_center_ex2, grid_total_width_ex2,
        grid_total_height_ex2, num_pixels_width_ex2, num_pixels_height_ex2,
        grid_angle_y_radians_ex2
    )
    print("Rasterized Grid (Example 2):\n", raster_ex2)

    # --- Example Usage 3: Line perpendicular to the plane (projects to a point) ---
    print("\n--- Example 3: Line perpendicular to the plane (projects to a point) ---")
    line_segment_ex3 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]], dtype=torch.float32)
    grid_center_ex3 = torch.tensor([0.0, 2.5, 0.0], dtype=torch.float32)
    grid_total_width_ex3 = 2.0
    grid_total_height_ex3 = 2.0
    num_pixels_width_ex3 = 2
    num_pixels_height_ex3 = 2
    grid_angle_y_radians_ex3 = 0.0 # Grid in XZ plane, normal is (0,1,0)

    raster_ex3 = rasterize_3d_line_segment_to_2d_grid(
        line_segment_ex3, grid_center_ex3, grid_total_width_ex3,
        grid_total_height_ex3, num_pixels_width_ex3, num_pixels_height_ex3,
        grid_angle_y_radians_ex3
    )
    print("Rasterized Grid (Example 3):\n", raster_ex3)
    # Expected: The entire length of the line (5.0) should be in one pixel if the projection point is within it.
    # Projected point is (0, 0, 0) in grid plane's 2D system if center is (0,2.5,0) and line is along Y.
    # If center is (0,2.5,0), line (0,0,0) to (0,5,0) with normal (0,1,0).
    # vec_to_start = (0,-2.5,0). dot(vec_to_start, u_vec=(1,0,0)) = 0.
    # dot(vec_to_start, v_vec=(0,0,1)) = 0. So projected point is (0,0) in 2D grid coords.
    # grid_min_u_coord = -1.0, grid_max_u_coord = 1.0
    # grid_min_v_coord = -1.0, grid_max_v_coord = 1.0
    # (0,0) is within the grid.
    # pixel_width_3d = 1.0, pixel_height_3d = 1.0
    # pixel_u_idx = int((0 - (-1))/1) = 1
    # pixel_v_idx = int((0 - (-1))/1) = 1
    # This means pixel (1,1) (bottom-right if 0-indexed from bottom-left)
    # For (height, width) with top-left origin: raster_grid[num_pixels_height - 1 - 1, 1] = raster_grid[0, 1] should be 5.0


    # --- Example Usage 4: Line partially outside the grid ---
    print("\n--- Example 4: Line partially outside the grid ---")
    line_segment_ex4 = torch.tensor([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32)
    grid_center_ex4 = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    grid_total_width_ex4 = 2.0
    grid_total_height_ex4 = 2.0
    num_pixels_width_ex4 = 2
    num_pixels_height_ex4 = 2
    grid_angle_y_radians_ex4 = 0.0

    raster_ex4 = rasterize_3d_line_segment_to_2d_grid(
        line_segment_ex4, grid_center_ex4, grid_total_width_ex4,
        grid_total_height_ex4, num_pixels_width_ex4, num_pixels_height_ex4,
        grid_angle_y_radians_ex4
    )
    print("Rasterized Grid (Example 4):\n", raster_ex4)
    # Expected: line from u=-2 to u=2. Grid from u=-1 to u=1.
    # The line is 4 units long. The part from -1 to 1 should be rasterized, length 2.0.
    # It should fall into the middle pixel(s) if grid is XZ.
    # Center (0,0,0), grid in XZ, line is on X-axis.
    # projected line from u=-2 to u=2, v=0.
    # grid_min_u = -1, grid_max_u = 1.
    # pixel_width = 1.0
    # pu_idx=0: [-1, 0]
    # pu_idx=1: [0, 1]
    # Line segment in grid: [-1, 1].
    # intersection with pu_idx=0: [-1, 0], length 1.0
    # intersection with pu_idx=1: [0, 1], length 1.0
    # Total length 2.0. Should be in one row, split between two pixels.
    # If it is the middle row (index 0 for 2x2 grid when flipping v_idx): [1.0, 1.0]
