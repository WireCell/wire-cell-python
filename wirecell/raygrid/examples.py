import math
import torch


def symmetric_views(width=100., height=100., pitch_mag=3,
                    angle=math.radians(60.0)):
    '''
    Return tensor suitable for giving to Coordinates().  Two wire directions
    are symmetric about the 3rd with givne angle.  First two views are
    horiz/vert bounds.

    This function is all in 2D with a (x,y) coordinate system.  If we consider
    looking in the direction of positive Z-axis then wire direction cross pitch
    direction is Z direction.  The layer directions:

    - pitch points toward U, wire points toward L (horiz bounds)
    - pitch points toward R, wire points toward U (vert bounds)
    - pitch points toward UL, wire points toward LL ("U" plane)
    - pitch points toward UR, wire points toward UL ("V" plane)
    - pitch points toward R, wire points toward U ("W" plane)

    Note, this convention is not directly WCT's usual which puts drift on X axis
    and may swap U/V depending on detector.
    '''
    pitches = torch.zeros((5, 2, 2))


    # horizontal ray bounds, pitch is vertical
    pitches[0] = torch.tensor([[width/2.0, 0], [width/2.0, height]])

    # vertical ray bounds, pitch is horizontal
    pitches[1] = torch.tensor([[0, height/2.0], [width, height/2.0]])

    
    # corners
    ll = torch.tensor([0,0])
    lr = torch.tensor([0,width])

    # /-wires as seen looking in negative-X direction
    w = torch.tensor([math.cos(angle), -math.sin(angle)])
    p = torch.tensor([-w[1], w[0]])
    ## inverse: w = [p[1], -p[0]]
    pitches[2] = torch.vstack([
        lr + 0.5*pitch_mag*p,
        lr + 1.5*pitch_mag*p
    ])

    # \-wires
    w = torch.tensor([math.cos(angle), +math.sin(angle)])
    p = torch.tensor([-w[1], w[0]])
    pitches[3] = torch.vstack([
        ll + 0.5*pitch_mag*p,
        ll + 1.5*pitch_mag*p
    ])

    # |-wires        
    pitches[4] = torch.tensor([[0, height/2.0], [pitch_mag, height/2.0]])

    return pitches



# gemini generated
import torch
import numpy as np

class Rasterizer3D:
    """
    A class to project and rasterize multiple 3D line segments onto a
    2D planar grid of pixels.

    The line is projected along the plane's normal direction. Each pixel on
    which the line is projected will have a value equal to the length of the
    3D element of the line that is covered by the pixel.
    """

    def __init__(self,
                 grid_center_3d: torch.Tensor,
                 grid_total_width: float,
                 grid_total_height: float,
                 num_pixels_width: int,
                 num_pixels_height: int,
                 grid_angle_y_radians: float):
        """
        Initializes the Rasterizer3D with the grid definition.

        Args:
            grid_center_3d (torch.Tensor): A PyTorch tensor of shape (3,) giving the
                                           3D point at which the center of the pixel
                                           grid rectangle is located.
            grid_total_width (float): The total width of the pixel grid rectangle
                                      (along the edge parallel to the X-axis).
            grid_total_height (float): The total height of the pixel grid rectangle.
            num_pixels_width (int): The number of pixels along the width.
            num_pixels_height (int): The number of pixels along the height.
            grid_angle_y_radians (float): The angle (in radians) of the grid plane
                                          with respect to the Y-axis, rotated around
                                          the X-axis.
        """
        if grid_center_3d.shape != (3,):
            raise ValueError("grid_center_3d must be a PyTorch tensor of shape (3,)")
        if not all(isinstance(arg, (int, float)) for arg in [grid_total_width, grid_total_height, num_pixels_width, num_pixels_height, grid_angle_y_radians]):
            raise TypeError("grid_total_width, grid_total_height, num_pixels_width, num_pixels_height, grid_angle_y_radians must be numeric")
        if num_pixels_width <= 0 or num_pixels_height <= 0:
            raise ValueError("num_pixels_width and num_pixels_height must be positive integers")

        self.grid_center_3d = grid_center_3d
        self.grid_total_width = grid_total_width
        self.grid_total_height = grid_total_height
        self.num_pixels_width = num_pixels_width
        self.num_pixels_height = num_pixels_height
        self.grid_angle_y_radians = grid_angle_y_radians

        # Initialize the raster grid (all zeros)
        self.raster_grid = torch.zeros((self.num_pixels_height, self.num_pixels_width), dtype=torch.float32)

        # Pre-calculate grid properties that are constant for all lines
        self._precompute_grid_properties()

    def _precompute_grid_properties(self):
        """
        Pre-calculates properties of the grid plane and pixel dimensions.
        """
        # The plane's normal vector.
        self.normal_vector = torch.tensor([
            0.0,
            math.cos(self.grid_angle_y_radians),
            math.sin(self.grid_angle_y_radians)
        ], dtype=torch.float32)
        self.normal_vector = self.normal_vector / torch.norm(self.normal_vector)

        # Basis vectors for the plane's 2D coordinate system
        self.u_vec = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        # Using torch.linalg.cross to avoid deprecation warning
        self.v_vec = torch.linalg.cross(self.normal_vector, self.u_vec)
        self.v_vec = self.v_vec / torch.norm(self.v_vec)

        # Calculate pixel dimensions
        self.pixel_width_3d = self.grid_total_width / self.num_pixels_width
        self.pixel_height_3d = self.grid_total_height / self.num_pixels_height

        # Define the 2D bounding box of the grid in the plane's coordinates
        self.grid_min_u_coord = -self.grid_total_width / 2.0
        self.grid_max_u_coord = self.grid_total_width / 2.0
        self.grid_min_v_coord = -self.grid_total_height / 2.0
        self.grid_max_v_coord = self.grid_total_height / 2.0

    def add_line(self, line_segment: torch.Tensor):
        """
        Adds a single 3D line segment to the raster.

        Args:
            line_segment (torch.Tensor): A PyTorch tensor of shape (2, 3) giving
                                         two 3D Cartesian points for the segment endpoints.
        """
        if line_segment.shape != (2, 3):
            raise ValueError("line_segment must be a PyTorch tensor of shape (2, 3)")

        # Project line segment endpoints onto the plane
        line_start_3d = line_segment[0]
        line_end_3d = line_segment[1]

        # Vector from grid center to line start point
        vec_to_start = line_start_3d - self.grid_center_3d
        # Vector from grid center to line end point
        vec_to_end = line_end_3d - self.grid_center_3d

        # Start and end points of the line in the plane's 2D coordinate system
        line_start_2d_u = torch.dot(vec_to_start, self.u_vec)
        line_start_2d_v = torch.dot(vec_to_start, self.v_vec)

        line_end_2d_u = torch.dot(vec_to_end, self.u_vec)
        line_end_2d_v = torch.dot(vec_to_end, self.v_vec)

        projected_line_start_2d = torch.tensor([line_start_2d_u, line_start_2d_v], dtype=torch.float32)
        projected_line_end_2d = torch.tensor([line_end_2d_u, line_end_2d_v], dtype=torch.float32)

        # 3D length of the original line segment
        original_line_length_3d = torch.norm(line_segment[1] - line_segment[0])

        # 2D length of the projected line segment
        projected_line_length_2d = torch.norm(projected_line_end_2d - projected_line_start_2d)

        EPS = 1e-9 # Epsilon for floating point comparisons

        if projected_line_length_2d < EPS:
            if original_line_length_3d > EPS: # Original line is not a point but projects to a point
                # Find the pixel this projected point falls into
                if (self.grid_min_u_coord <= projected_line_start_2d[0] <= self.grid_max_u_coord and
                    self.grid_min_v_coord <= projected_line_start_2d[1] <= self.grid_max_v_coord):

                    pixel_u_idx = int(torch.floor((projected_line_start_2d[0] - self.grid_min_u_coord) / self.pixel_width_3d))
                    pixel_v_idx = int(torch.floor((projected_line_start_2d[1] - self.grid_min_v_coord) / self.pixel_height_3d))

                    pixel_u_idx = max(0, min(pixel_u_idx, self.num_pixels_width - 1))
                    pixel_v_idx = max(0, min(pixel_v_idx, self.num_pixels_height - 1))

                    # Add the length to the raster grid.
                    # Remember: num_pixels_height - 1 - pixel_v_idx for top-left origin.
                    self.raster_grid[self.num_pixels_height - 1 - pixel_v_idx, pixel_u_idx] += original_line_length_3d
            return # No line or only a point to raster

        length_ratio_3d_to_2d = original_line_length_3d / projected_line_length_2d

        # Determine min/max pixel indices that the projected line could possibly cover
        min_u_proj = min(projected_line_start_2d[0], projected_line_end_2d[0])
        max_u_proj = max(projected_line_start_2d[0], projected_line_end_2d[0])
        min_v_proj = min(projected_line_start_2d[1], projected_line_end_2d[1])
        max_v_proj = max(projected_line_start_2d[1], projected_line_end_2d[1])

        # Clamp these to the grid boundaries
        start_pixel_u = int(torch.floor((min_u_proj - self.grid_min_u_coord) / self.pixel_width_3d).clamp(0, self.num_pixels_width - 1))
        end_pixel_u = int(torch.ceil((max_u_proj - self.grid_min_u_coord) / self.pixel_width_3d).clamp(0, self.num_pixels_width - 1))

        start_pixel_v = int(torch.floor((min_v_proj - self.grid_min_v_coord) / self.pixel_height_3d).clamp(0, self.num_pixels_height - 1))
        end_pixel_v = int(torch.ceil((max_v_proj - self.grid_min_v_coord) / self.pixel_height_3d).clamp(0, self.num_pixels_height - 1))

        line_dir = projected_line_end_2d - projected_line_start_2d

        # Iterate over relevant pixels
        for pv_idx in range(start_pixel_v, end_pixel_v + 1):
            for pu_idx in range(start_pixel_u, end_pixel_u + 1):
                # Calculate pixel's 2D bounding box in the plane's coordinate system
                pixel_min_u = self.grid_min_u_coord + pu_idx * self.pixel_width_3d
                pixel_max_u = self.grid_min_u_coord + (pu_idx + 1) * self.pixel_width_3d
                pixel_min_v = self.grid_min_v_coord + pv_idx * self.pixel_height_3d
                pixel_max_v = self.grid_min_v_coord + (pv_idx + 1) * self.pixel_height_3d

                t_values = []

                # Check intersection with vertical lines of pixel box (u-bounds)
                if abs(line_dir[0]) > EPS:
                    t1 = (pixel_min_u - projected_line_start_2d[0]) / line_dir[0]
                    t2 = (pixel_max_u - projected_line_start_2d[0]) / line_dir[0]
                    t_values.extend([t1, t2])

                # Check intersection with horizontal lines of pixel box (v-bounds)
                if abs(line_dir[1]) > EPS:
                    t3 = (pixel_min_v - projected_line_start_2d[1]) / line_dir[1]
                    t4 = (pixel_max_v - projected_line_start_2d[1]) / line_dir[1]
                    t_values.extend([t3, t4])

                # Include t=0 and t=1 for segment endpoints
                t_values.extend([0.0, 1.0])

                # Filter valid t values (0 <= t <= 1) and sort them
                t_values = [t for t in t_values if 0.0 - EPS <= t <= 1.0 + EPS] # Add epsilon for robustness
                t_values = sorted(list(set(t_values)))

                if len(t_values) < 2:
                    continue

                segment_length_in_pixel_2d = 0.0
                for i in range(len(t_values) - 1):
                    t_start = t_values[i]
                    t_end = t_values[i+1]

                    t_mid = (t_start + t_end) / 2.0
                    mid_point_2d = projected_line_start_2d + t_mid * line_dir

                    if (pixel_min_u <= mid_point_2d[0] <= pixel_max_u and
                        pixel_min_v <= mid_point_2d[1] <= pixel_max_v):

                        sub_segment_start_2d = projected_line_start_2d + t_start * line_dir
                        sub_segment_end_2d = projected_line_start_2d + t_end * line_dir
                        segment_length_in_pixel_2d += torch.norm(sub_segment_end_2d - sub_segment_start_2d)

                if segment_length_in_pixel_2d > EPS:
                    length_3d_in_pixel = segment_length_in_pixel_2d * length_ratio_3d_to_2d
                    # Add to the raster grid (accumulate)
                    self.raster_grid[self.num_pixels_height - 1 - pv_idx, pu_idx] += length_3d_in_pixel

    def get_raster(self) -> torch.Tensor:
        """
        Returns the current accumulated raster grid.

        Returns:
            torch.Tensor: The 2D PyTorch array representing the rasterized lines.
        """
        return self.raster_grid


def rasterize_3d_line_segment_to_2d_grid(
    line_segment: torch.Tensor,
    grid_center_3d: torch.Tensor,
    grid_total_width: float,
    grid_total_height: float,
    num_pixels_width: int,
    num_pixels_height: int,
    grid_angle_y_radians: float # Angle with respect to the Y-axis (around X-axis)
) -> torch.Tensor:
    """
    Projects and rasterizes a 3D line segment onto a 2D planar grid of pixels.

    The line is projected along the plane's normal direction. Each pixel on
    which the line is projected will have a value equal to the length of the
    3D element of the line that is covered by the pixel.

    Args:
        line_segment (torch.Tensor): A PyTorch tensor of shape (2, 3) giving
                                     two 3D Cartesian points for the segment endpoints.
        grid_center_3d (torch.Tensor): A PyTorch tensor of shape (3,) giving the
                                       3D point at which the center of the pixel
                                       grid rectangle is located.
        grid_total_width (float): The total width of the pixel grid rectangle
                                  (along the edge parallel to the X-axis).
        grid_total_height (float): The total height of the pixel grid rectangle.
        num_pixels_width (int): The number of pixels along the width.
        num_pixels_height (int): The number of pixels along the height.
        grid_angle_y_radians (float): The angle (in radians) of the grid plane
                                      with respect to the Y-axis, rotated around
                                      the X-axis.

    Returns:
        torch.Tensor: A 2D PyTorch array of shape (num_pixels_height, num_pixels_width)
                      representing the rasterized line, where each element is the
                      length of the 3D line segment portion covered by that pixel.
    """
    if line_segment.shape != (2, 3):
        raise ValueError("line_segment must be a PyTorch tensor of shape (2, 3)")
    if grid_center_3d.shape != (3,):
        raise ValueError("grid_center_3d must be a PyTorch tensor of shape (3,)")
    if not all(isinstance(arg, (int, float)) for arg in [grid_total_width, grid_total_height, num_pixels_width, num_pixels_height, grid_angle_y_radians]):
        raise TypeError("grid_total_width, grid_total_height, num_pixels_width, num_pixels_height, grid_angle_y_radians must be numeric")
    if num_pixels_width <= 0 or num_pixels_height <= 0:
        raise ValueError("num_pixels_width and num_pixels_height must be positive integers")

    # 1. Define the Pixel Grid in 3D and its coordinate system

    # The plane's normal vector. Since one edge is parallel to the X-axis,
    # and it's rotated around the X-axis with respect to the Y-axis,
    # the normal will be in the YZ-plane.
    # If angle is 0, normal is (0, 1, 0) (plane is XZ plane)
    # If angle is pi/2, normal is (0, 0, 1) (plane is XY plane)
    normal_vector = torch.tensor([
        0.0,
        math.cos(grid_angle_y_radians),
        math.sin(grid_angle_y_radians)
    ], dtype=torch.float32)
    normal_vector = normal_vector / torch.norm(normal_vector) # Ensure unit vector

    # Basis vectors for the plane's 2D coordinate system
    # u_vec is along the width, parallel to X-axis
    u_vec = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    # v_vec is along the height, orthogonal to u_vec and in the plane
    v_vec = torch.cross(normal_vector, u_vec, dim=-1)
    v_vec = v_vec / torch.norm(v_vec) # Ensure unit vector

    # Check for potential issues if normal_vector is parallel to u_vec (not expected with current setup)
    if torch.dot(normal_vector, u_vec).abs() > 1e-6:
        print("Warning: Normal vector not orthogonal to X-axis direction. Check angle definition.")

    # Calculate pixel dimensions
    pixel_width_3d = grid_total_width / num_pixels_width
    pixel_height_3d = grid_total_height / num_pixels_height

    # Rasterization grid initialization
    raster_grid = torch.zeros((num_pixels_height, num_pixels_width), dtype=torch.float32)

    # Project line segment endpoints onto the plane
    line_start_3d = line_segment[0]
    line_end_3d = line_segment[1]

    # Vector from grid center to line start point
    vec_to_start = line_start_3d - grid_center_3d
    # Vector from grid center to line end point
    vec_to_end = line_end_3d - grid_center_3d

    # Project the vectors onto the plane and get their 2D coordinates
    # The projection is done along the normal.
    # A point P's projection P_proj onto a plane with normal N and point C is P - dot(P-C, N) * N
    # Since we defined our plane coordinate system based on grid_center_3d as origin
    # the 2D coordinates (u,v) for a point P_3d are:
    # u = dot(P_3d - grid_center_3d, u_vec)
    # v = dot(P_3d - grid_center_3d, v_vec)

    # Start and end points of the line in the plane's 2D coordinate system
    # We define the origin of this 2D system at the grid_center_3d
    # and the +u direction along u_vec, +v along v_vec.
    # The grid's bottom-left corner in this 2D system will be (-grid_total_width/2, -grid_total_height/2)
    line_start_2d_u = torch.dot(vec_to_start, u_vec)
    line_start_2d_v = torch.dot(vec_to_start, v_vec)

    line_end_2d_u = torch.dot(vec_to_end, u_vec)
    line_end_2d_v = torch.dot(vec_to_end, v_vec)

    projected_line_start_2d = torch.tensor([line_start_2d_u, line_start_2d_v], dtype=torch.float32)
    projected_line_end_2d = torch.tensor([line_end_2d_u, line_end_2d_v], dtype=torch.float32)

    # 3D length of the original line segment
    original_line_length_3d = torch.norm(line_segment[1] - line_segment[0])

    # 2D length of the projected line segment
    projected_line_length_2d = torch.norm(projected_line_end_2d - projected_line_start_2d)

    # Ratio of 3D length to 2D length (for scaling back later)
    # Handle cases where projected_line_length_2d is zero (line is perpendicular to plane)
    if projected_line_length_2d < 1e-6:
        if original_line_length_3d < 1e-6: # Original line is a point
            return raster_grid # No line to raster
        else: # Original line is not a point but projects to a point
            # Find the pixel this projected point falls into
            # First, check if the projected point is within the grid bounds
            grid_min_u = -grid_total_width / 2.0
            grid_max_u = grid_total_width / 2.0
            grid_min_v = -grid_total_height / 2.0
            grid_max_v = grid_total_height / 2.0

            if (grid_min_u <= projected_line_start_2d[0] <= grid_max_u and
                grid_min_v <= projected_line_start_2d[1] <= grid_max_v):

                # Calculate pixel indices
                pixel_u_idx = int((projected_line_start_2d[0] - grid_min_u) / pixel_width_3d)
                pixel_v_idx = int((projected_line_start_2d[1] - grid_min_v) / pixel_height_3d)

                # Clamp indices to valid range (shouldn't be necessary if bounds check is good, but for safety)
                pixel_u_idx = max(0, min(pixel_u_idx, num_pixels_width - 1))
                pixel_v_idx = max(0, min(pixel_v_idx, num_pixels_height - 1))

                # Note: PyTorch arrays are (height, width) i.e. (rows, cols) where row is v-axis, col is u-axis
                # So pixel_v_idx corresponds to row, pixel_u_idx to col
                # For image coordinates, usually (0,0) is top-left, so we might need to invert v_idx
                # For now, let's assume (0,0) is bottom-left for simplicity in geometric calculations.
                # If you need top-left, adjust the v_idx: num_pixels_height - 1 - pixel_v_idx
                raster_grid[num_pixels_height - 1 - pixel_v_idx, pixel_u_idx] = original_line_length_3d
            return raster_grid


    length_ratio_3d_to_2d = original_line_length_3d / projected_line_length_2d

    # Define the 2D bounding box of the grid in the plane's coordinates
    grid_min_u_coord = -grid_total_width / 2.0
    grid_max_u_coord = grid_total_width / 2.0
    grid_min_v_coord = -grid_total_height / 2.0
    grid_max_v_coord = grid_total_height / 2.0

    # Iterate over pixels (using a simple DDA-like approach or Bresenham variant)
    # Or, iterate over the projected line segment and find intersecting pixels.
    # A more robust approach for line-pixel intersection:
    # 1. Determine the bounding box of the projected line segment in 2D.
    # 2. Iterate over pixels within this bounding box (and grid bounds).
    # 3. For each pixel, check if the line segment intersects the pixel's bounding box.
    # 4. If it intersects, calculate the segment of the line within the pixel.

    # Determine min/max pixel indices that the projected line could possibly cover
    min_u_proj = min(projected_line_start_2d[0], projected_line_end_2d[0])
    max_u_proj = max(projected_line_start_2d[0], projected_line_end_2d[0])
    min_v_proj = min(projected_line_start_2d[1], projected_line_end_2d[1])
    max_v_proj = max(projected_line_start_2d[1], projected_line_end_2d[1])

    # Convert projected 2D coordinates to pixel indices range
    # U-axis: maps from [-grid_total_width/2, grid_total_width/2] to [0, num_pixels_width]
    # V-axis: maps from [-grid_total_height/2, grid_total_height/2] to [0, num_pixels_height]

    # Start pixel indices (clamped to grid)
    start_pixel_u = int(torch.floor((min_u_proj - grid_min_u_coord) / pixel_width_3d).clamp(0, num_pixels_width - 1))
    end_pixel_u = int(torch.ceil((max_u_proj - grid_min_u_coord) / pixel_width_3d).clamp(0, num_pixels_width - 1))

    start_pixel_v = int(torch.floor((min_v_proj - grid_min_v_coord) / pixel_height_3d).clamp(0, num_pixels_height - 1))
    end_pixel_v = int(torch.ceil((max_v_proj - grid_min_v_coord) / pixel_height_3d).clamp(0, num_pixels_height - 1))

    # Iterate over relevant pixels
    for pv_idx in range(start_pixel_v, end_pixel_v + 1):
        for pu_idx in range(start_pixel_u, end_pixel_u + 1):
            # Calculate pixel's 2D bounding box in the plane's coordinate system
            # Note: pu_idx and pv_idx are 0-indexed.
            # (0,0) is bottom-left pixel, corresponds to (grid_min_u_coord, grid_min_v_coord)
            pixel_min_u = grid_min_u_coord + pu_idx * pixel_width_3d
            pixel_max_u = grid_min_u_coord + (pu_idx + 1) * pixel_width_3d
            pixel_min_v = grid_min_v_coord + pv_idx * pixel_height_3d
            pixel_max_v = grid_min_v_coord + (pv_idx + 1) * pixel_height_3d

            # Intersection of line segment with pixel bounding box (using clipping algorithm like Liang-Barsky or Cohen-Sutherland)
            # For simplicity, we'll use a parametric line intersection check.
            # Line: P(t) = P_start + t * (P_end - P_start), 0 <= t <= 1
            # Pixel bounds: u_min <= P(t)_u <= u_max, v_min <= P(t)_v <= v_max

            line_dir = projected_line_end_2d - projected_line_start_2d
            # Avoid division by zero for horizontal/vertical lines
            EPS = 1e-9

            t_values = []

            # Check intersection with vertical lines of pixel box (u-bounds)
            if abs(line_dir[0]) > EPS:
                t1 = (pixel_min_u - projected_line_start_2d[0]) / line_dir[0]
                t2 = (pixel_max_u - projected_line_start_2d[0]) / line_dir[0]
                t_values.extend([t1, t2])

            # Check intersection with horizontal lines of pixel box (v-bounds)
            if abs(line_dir[1]) > EPS:
                t3 = (pixel_min_v - projected_line_start_2d[1]) / line_dir[1]
                t4 = (pixel_max_v - projected_line_start_2d[1]) / line_dir[1]
                t_values.extend([t3, t4])

            # Include t=0 and t=1 for segment endpoints
            t_values.extend([0.0, 1.0])

            # Filter valid t values (0 <= t <= 1) and sort them
            t_values = [t for t in t_values if 0.0 <= t <= 1.0]
            t_values = sorted(list(set(t_values))) # Use set to remove duplicates

            if len(t_values) < 2:
                continue # No segment or only a point within the current pixel

            segment_length_in_pixel_2d = 0.0
            for i in range(len(t_values) - 1):
                t_start = t_values[i]
                t_end = t_values[i+1]

                # Midpoint of the potential sub-segment for checking inclusion
                t_mid = (t_start + t_end) / 2.0
                mid_point_2d = projected_line_start_2d + t_mid * line_dir

                # Check if this sub-segment's midpoint is inside the current pixel
                if (pixel_min_u <= mid_point_2d[0] <= pixel_max_u and
                    pixel_min_v <= mid_point_2d[1] <= pixel_max_v):
                    
                    # Calculate the length of this 2D sub-segment
                    sub_segment_start_2d = projected_line_start_2d + t_start * line_dir
                    sub_segment_end_2d = projected_line_start_2d + t_end * line_dir
                    segment_length_in_pixel_2d += torch.norm(sub_segment_end_2d - sub_segment_start_2d)

            if segment_length_in_pixel_2d > EPS: # Only add if there's a significant length
                # Convert 2D length back to 3D length using the ratio
                length_3d_in_pixel = segment_length_in_pixel_2d * length_ratio_3d_to_2d

                # Assign to raster grid. Remember to flip v_idx for image-like (top-left origin) indexing
                raster_grid[num_pixels_height - 1 - pv_idx, pu_idx] = length_3d_in_pixel

    return raster_grid

