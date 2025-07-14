import math
import torch
import numpy as np


def symmetric_views(width=100., height=100., pitch_mag=3,
                    angle=math.radians(60.0)):
    '''
    Return a (5,2,2) pitch ray tensor suitable for giving to Coordinates().

    Angle is from y-axis to wire direction.

    This function is equivalent to symmetric_raypairs() in WCT's
    util/src/RayHelpers.cxx except it returns a pitch ray instead of a pair of
    wire rays and it operates in 2D (x_rg, y_rg) instead of 3D (x_wc, y_wc,
    z_wc).  The coordinates correspondence is:

    y_rg == y_wc
    x_rg == z_wc
    z_rg == -x_wc

    Note, z_rg is not used here but may be defined by right-hand-rule.  Also by
    RHR, "pitch cross wire = z_rg" (WCT it's opposite: "wire cross pitch = x_wc").

    Rays generally start on left hand side, pitch increasing toward right
    (x_rg).  This is equivalent to increasing with z_wc.  Special case
    horizontal rays start at bottom and go up.
    '''
    pitches = torch.zeros((5, 2, 2))

    # horizontal ray bounds, pitch is vertical
    pitches[0] = torch.tensor([[width/2.0, 0], [width/2.0, height]])

    # vertical ray bounds, pitch is horizontal
    pitches[1] = torch.tensor([[0, height/2.0], [width, height/2.0]])

    # corners
    ll = torch.tensor([0,0])
    ul = torch.tensor([0,height])
    lr = torch.tensor([width,0])

    # /-wires
    w = torch.tensor([math.sin(angle), math.cos(angle)])
    p = torch.tensor([w[1], -w[0]])
    pitches[2] = torch.vstack([
        ul + 0.5*pitch_mag*p,
        ul + 1.5*pitch_mag*p
    ])

    # the symmetry
    angle *= -1

    # \-wires
    w = torch.tensor([math.sin(angle), math.cos(angle)])
    p = torch.tensor([w[1], -w[0]])
    pitches[3] = torch.vstack([
        ll + 0.5*pitch_mag*p,
        ll + 1.5*pitch_mag*p
    ])

    # |-wires        
    pitches[4] = torch.tensor([[0, height/2.0], [pitch_mag, height/2.0]])

    return pitches

def random_points(npoints=100, ll=(0.,0.), ur=(100.0,100.0)):
    '''
    Return points randomly chosen in a rectangular domain
    '''
    points = torch.rand(npoints, 2, dtype=torch.float32)
    for dim in [0,1]:
        points[:, dim] = points[:, dim] * (ur[dim]-ll[dim]) + ll[dim]
    return points


def random_groups(ngroups: int = 10, n_in_group: int = 10, sigma: float = 10,
                  ll: tuple[float, float] = (0.0, 0.0),
                  ur: tuple[float, float] = (100.0, 100.0)) -> torch.Tensor:
    """
    Generates "clumpy" 2D points by first creating group centers uniformly
    and then sampling points around these centers from a Gaussian distribution.

    Args:
        ngroups: The number of distinct groups (clusters) of points.
        n_in_group: The number of points to generate within each group.
        sigma: The standard deviation (width) of the Gaussian distribution
               for points within each group.
        ll: A tuple (x_min, y_min) representing the lower-left corner
            of the bounding box for generating group centers.
        ur: A tuple (x_max, y_max) representing the upper-right corner
            of the bounding box for generating group centers.

    Returns:
        A torch.Tensor of shape (ngroups, n_in_group, 2) containing
        the generated 2D points. Points outside the bounding box are
        included as per the requirement.
    """
    # 1. Generate `ngroups` "group points" uniformly random within the bounding box.
    # These will serve as the mean locations for the Gaussian distributions.
    x_range = ur[0] - ll[0]
    y_range = ur[1] - ll[1]

    # Generate points in [0, 1) and then scale/shift
    group_means = torch.rand(ngroups, 2, dtype=torch.float32)
    group_means[:, 0] = group_means[:, 0] * x_range + ll[0]
    group_means[:, 1] = group_means[:, 1] * y_range + ll[1]

    # 2. For each group point, generate `n_in_group` points
    # according to a Gaussian distribution centered at the group point.

    # Generate standard normal noise: (ngroups, n_in_group, 2)
    # Each row (ngroups) corresponds to a group, each column (n_in_group) to a point in that group,
    # and the last dim (2) for X, Y coordinates.
    noise = torch.randn(ngroups, n_in_group, 2, dtype=torch.float32)

    # Scale the noise by sigma (standard deviation)
    scaled_noise = noise * sigma

    # Add the group means to the scaled noise.
    # `group_means` is (ngroups, 2). To add it to `scaled_noise` (ngroups, n_in_group, 2),
    # we need to expand `group_means` to (ngroups, 1, 2) to enable broadcasting.
    # The '1' in the middle dimension will broadcast across 'n_in_group'.
    points_in_groups = group_means.unsqueeze(1) + scaled_noise

    return points_in_groups

def fill_activity(coords, points):
    nviews = coords.nviews

    activity_tensors = []

    # Get all pitch indices for all points across all views in one vectorized call
    all_pitch_indices = coords.point_indices(points)  # Shape (num_points, nviews)

    # Iterate through each view to fill its individual activity tensor
    for view_idx in range(nviews):
        current_view_indices = all_pitch_indices[:, view_idx]

        # Handle the special cases for the first two views (view_idx 0 and 1)
        if view_idx in [0, 1]:
            # The requirement is for activity tensors of size 1 for these views.
            # This implies a simple boolean state: True if any point maps to a valid index (>=0), False otherwise.
            activity_tensor = torch.zeros(1, dtype=torch.bool, device=points.device)
            if (current_view_indices == 0).any(): # Check if at least one point has a non-negative index
                activity_tensor[0] = True
            activity_tensors.append(activity_tensor)

        else:
            # For other views (view_idx >= 2):
            # Calculate the minimum and maximum pitch indices encountered for this view.
            min_idx = current_view_indices.min().item()
            max_idx = current_view_indices.max().item()

            # Adjust indices to be non-negative if they contain negative values.
            # This offset will shift all indices so that the smallest index becomes 0.
            offset = 0
            if min_idx < 0:
                offset = -min_idx

            adjusted_indices = current_view_indices + offset

            # Determine the required size for the activity tensor based on the adjusted indices.
            # It should cover from 0 up to the maximum adjusted index.
            effective_max_idx = adjusted_indices.max().item()
            required_size = effective_max_idx + 1

            # Apply the constraint: activity tensors should not be larger than 100 elements.
            # We cap the size at 100 if the required size is greater.
            activity_tensor_size = min(required_size, 100)

            # Create the activity tensor initialized to False
            activity_tensor = torch.zeros(activity_tensor_size, dtype=torch.bool, device=points.device)

            # Clamp the adjusted indices to fit within the `activity_tensor_size`.
            # Any index that would fall outside the capped size will be mapped to the boundary.
            adjusted_indices_clamped = torch.clamp(adjusted_indices, 0, activity_tensor_size - 1)

            # Set the corresponding positions in the activity tensor to True
            activity_tensor[adjusted_indices_clamped] = True
            activity_tensors.append(activity_tensor)
    return activity_tensors






class Raster:
    """
    Represents a 3D to 2D projection of 3D line segments onto a finite 2D rectangle
    divided into pixels. Each pixel accumulates the length of the projected line
    segment that falls onto it.
    """

    def __init__(self, normal, center, direction, size, shape):
        """
        Initializes the Raster object with the rectangle's properties.

        Args:
            normal (torch.Tensor): A 1D tensor of 3 elements representing the
                                   3D vector perpendicular to the plane of the rectangle.
            center (torch.Tensor): A 1D tensor of 3 elements representing the
                                   3D point at the center of the rectangle.
            direction (torch.Tensor): A 1D tensor of 3 elements representing the
                                      3D vector along the length dimension of the rectangle.
            size (tuple): A pair of real numbers (length, width) giving the dimensions
                          of the rectangle. 'length' is along 'direction', 'width' is transverse.
            shape (tuple): An integer pair (num_pixels_length, num_pixels_width) giving
                           the number of pixels in each dimension, corresponding to 'size'.
        
        Raises:
            ValueError: If input tensors have incorrect shapes or types, or if
                        'direction' is parallel to 'normal'.
        """
        # Input validation
        if not (isinstance(normal, torch.Tensor) and normal.shape == (3,)):
            raise ValueError("normal must be a 1D tensor of 3 elements.")
        if not (isinstance(center, torch.Tensor) and center.shape == (3,)):
            raise ValueError("center must be a 1D tensor of 3 elements.")
        if not (isinstance(direction, torch.Tensor) and direction.shape == (3,)):
            raise ValueError("direction must be a 1D tensor of 3 elements.")
        if not (isinstance(size, tuple) and len(size) == 2 and all(isinstance(s, (int, float)) for s in size)):
            raise ValueError("size must be a tuple of two numbers (length, width).")
        if not (isinstance(shape, tuple) and len(shape) == 2 and all(isinstance(s, int) and s > 0 for s in shape)):
            raise ValueError("shape must be a tuple of two positive integers (num_pixels_length, num_pixels_width).")

        # Store input parameters, ensuring float type for calculations
        self.normal = normal.float()
        self.center = center.float()
        self.direction = direction.float()
        self.size = size
        self.shape = shape

        # Normalize the normal vector
        self.normal_unit = torch.nn.functional.normalize(self.normal, dim=0)

        # Project the 'direction' vector onto the plane and normalize to get u_vec
        # This ensures u_vec is truly in the plane and orthogonal to normal_unit
        direction_proj = self.direction - torch.dot(self.direction, self.normal_unit) * self.normal_unit
        
        # Handle edge case where original 'direction' is almost parallel to 'normal'
        if torch.norm(direction_proj) < 1e-6:
            # If direction is parallel to normal, pick an arbitrary orthogonal direction in the plane.
            # Try (1,0,0), then (0,1,0), then (0,0,1) until a non-parallel vector is found.
            temp_vecs = [
                torch.tensor([1.0, 0.0, 0.0], dtype=torch.float),
                torch.tensor([0.0, 1.0, 0.0], dtype=torch.float),
                torch.tensor([0.0, 0.0, 1.0], dtype=torch.float)
            ]
            found_valid_direction = False
            for temp_vec in temp_vecs:
                direction_proj = temp_vec - torch.dot(temp_vec, self.normal_unit) * self.normal_unit
                if torch.norm(direction_proj) > 1e-6:
                    found_valid_direction = True
                    break
            if not found_valid_direction:
                raise ValueError("Could not determine a valid 'direction' vector in the plane. "
                                 "This might happen if 'normal' is degenerate or if all axis vectors are parallel to it.")

        self.u_vec = torch.nn.functional.normalize(direction_proj, dim=0)
        
        # Calculate v_vec (transverse direction) using cross product
        # v_vec = normal_unit x u_vec ensures a right-handed coordinate system in the plane
        self.v_vec = torch.nn.functional.normalize(torch.linalg.cross(self.normal_unit, self.u_vec), dim=0)

        # Calculate the 3D coordinates of the rectangle's origin (bottom-left corner in local coordinates)
        self.rect_origin_3d = self.center - (self.size[0] / 2) * self.u_vec - (self.size[1] / 2) * self.v_vec

        # Calculate the physical size of each pixel
        self.pixel_size_x = self.size[0] / self.shape[0] # Length per pixel
        self.pixel_size_y = self.size[1] / self.shape[1] # Width per pixel

        # Initialize the 2D pixel grid with zeros.
        # The shape is (num_pixels_width, num_pixels_length) to correspond to (rows, columns).
        self.pixels = torch.zeros((self.shape[1], self.shape[0]), dtype=torch.float)

    def _project_point_to_plane(self, point):
        """
        Projects a 3D point onto the plane defined by self.normal_unit and self.center.

        Args:
            point (torch.Tensor): A 1D tensor of 3 elements representing the 3D point.

        Returns:
            torch.Tensor: The 3D projected point on the plane.
        """
        # Vector from the plane's center to the point
        vec_to_point = point - self.center
        # Distance from the point to the plane along the normal vector
        dist = torch.dot(vec_to_point, self.normal_unit)
        # The projected point is found by moving 'dist' along the negative normal from the original point
        return point - dist * self.normal_unit

    def _to_local_2d(self, point_3d_proj):
        """
        Converts a 3D point (which is assumed to be on the plane) to its 2D local
        coordinates relative to the rectangle's origin (bottom-left corner).

        Args:
            point_3d_proj (torch.Tensor): A 1D tensor of 3 elements representing the
                                          3D point projected onto the plane.

        Returns:
            torch.Tensor: A 1D tensor of 2 elements (x_local, y_local) in the
                          rectangle's local coordinate system.
        """
        # Vector from the rectangle's 3D origin to the projected 3D point
        vec_from_rect_origin = point_3d_proj - self.rect_origin_3d
        # Project this vector onto u_vec and v_vec to get local 2D coordinates
        x_local = torch.dot(vec_from_rect_origin, self.u_vec)
        y_local = torch.dot(vec_from_rect_origin, self.v_vec)
        return torch.tensor([x_local, y_local], dtype=torch.float)

    def _liang_barsky_clip(self, p1, p2):
        """
        Applies the Liang-Barsky line clipping algorithm to a 2D line segment.
        Clips the line segment (p1, p2) against the rectangle defined by
        [0, self.size[0]] for x and [0, self.size[1]] for y.

        Args:
            p1 (torch.Tensor): A 1D tensor of 2 elements representing the first
                               endpoint of the 2D line segment.
            p2 (torch.Tensor): A 1D tensor of 2 elements representing the second
                               endpoint of the 2D line segment.

        Returns:
            tuple or None: A tuple (clipped_p1, clipped_p2) of 1D tensors if the
                           segment intersects the rectangle, otherwise None.
        """
        # Define the clipping window boundaries
        xmin, ymin = 0.0, 0.0
        xmax, ymax = float(self.size[0]), float(self.size[1])

        # Calculate line segment direction vector components
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        # Parameters for Liang-Barsky algorithm
        # p values: [-dx, dx, -dy, dy]
        # q values: [x1 - xmin, xmax - x1, y1 - ymin, ymax - y1]
        p = torch.tensor([-dx, dx, -dy, dy], dtype=torch.float)
        q = torch.tensor([p1[0] - xmin, xmax - p1[0], p1[1] - ymin, ymax - p1[1]], dtype=torch.float)

        u1 = 0.0 # Parameter for the entering point of the clipped segment
        u2 = 1.0 # Parameter for the exiting point of the clipped segment

        for k in range(4): # Iterate through the four clipping boundaries
            if torch.abs(p[k]) < 1e-9: # Line is parallel to this clipping edge
                if q[k] < 0: # And it's outside the boundary
                    return None # No intersection
            else:
                t = q[k] / p[k] # Calculate intersection parameter
                if p[k] < 0: # Entering point (from outside to inside)
                    u1 = max(u1, t.item())
                else: # Exiting point (from inside to outside)
                    u2 = min(u2, t.item())
        
        if u1 > u2:
            return None # No intersection (segment is entirely outside or crosses and exits before entering)

        # Calculate the clipped endpoints using the determined parameters u1 and u2
        clipped_p1 = p1 + u1 * (p2 - p1)
        clipped_p2 = p1 + u2 * (p2 - p1)
        return clipped_p1, clipped_p2

    def add_line(self, tail, head):
        """
        Projects a 3D line segment (tail, head) onto the rectangle and updates
        the pixel values in the internal `pixels` tensor. The value added to
        each pixel is proportional to the length of the projected segment
        that falls onto that pixel.

        Args:
            tail (torch.Tensor): A 1D tensor of 3 elements representing the
                                 first endpoint of the 3D line segment.
            head (torch.Tensor): A 1D tensor of 3 elements representing the
                                 second endpoint of the 3D line segment.
        
        Raises:
            ValueError: If input tensors for tail or head have incorrect shapes.
        """
        # Input validation
        if not (isinstance(tail, torch.Tensor) and tail.shape == (3,)):
            raise ValueError("tail must be a 1D tensor of 3 elements.")
        if not (isinstance(head, torch.Tensor) and head.shape == (3,)):
            raise ValueError("head must be a 1D tensor of 3 elements.")
        
        # Ensure float type for calculations
        tail = tail.float()
        head = head.float()

        # 1. Project 3D line segment endpoints onto the plane
        tail_proj_3d = self._project_point_to_plane(tail)
        head_proj_3d = self._project_point_to_plane(head)

        # If the projected line segment collapses to a single point (e.g., original line
        # was parallel to the normal and projected onto a single point on the plane)
        if torch.norm(tail_proj_3d - head_proj_3d) < 1e-9:
            point_2d = self._to_local_2d(tail_proj_3d)
            
            # Check if this single projected point is within the rectangle bounds
            if 0 <= point_2d[0] < self.size[0] and 0 <= point_2d[1] < self.size[1]:
                # Convert local 2D coordinates to pixel indices
                col_idx = int(point_2d[0] / self.pixel_size_x)
                row_idx = int(point_2d[1] / self.pixel_size_y)
                
                # Ensure indices are within the valid range of the pixel grid
                col_idx = min(max(0, col_idx), self.shape[0] - 1)
                row_idx = min(max(0, row_idx), self.shape[1] - 1)
                
                # Add a small fixed value for a point projection (e.g., 1.0)
                self.pixels[row_idx, col_idx] += 1.0 
            return

        # 2. Convert projected 3D points to 2D local coordinates within the rectangle's plane
        tail_proj_2d = self._to_local_2d(tail_proj_3d)
        head_proj_2d = self._to_local_2d(head_proj_3d)

        # 3. Clip the 2D line segment against the rectangle's boundaries
        clipped_segment = self._liang_barsky_clip(tail_proj_2d, head_proj_2d)

        if clipped_segment is None:
            return # The line segment is entirely outside the rectangle, so do nothing

        clipped_p1_2d, clipped_p2_2d = clipped_segment

        # 4. Rasterize the clipped 2D segment onto the pixel grid
        # Calculate the length of the clipped 2D segment
        segment_length_2d = torch.norm(clipped_p2_2d - clipped_p1_2d).item()

        # If the clipped segment is extremely short, treat it as a point
        if segment_length_2d < 1e-9:
            point_2d = clipped_p1_2d # Use one of the clipped points
            
            # Convert local 2D coordinates to pixel indices
            col_idx = int(point_2d[0] / self.pixel_size_x)
            row_idx = int(point_2d[1] / self.pixel_size_y)
            
            # Ensure indices are within bounds
            col_idx = min(max(0, col_idx), self.shape[0] - 1)
            row_idx = min(max(0, row_idx), self.shape[1] - 1)
            
            self.pixels[row_idx, col_idx] += 1.0 # Add a small fixed value
            return

        # Determine the number of steps for rasterization based on pixel resolution
        # This ensures that we sample enough points to cover all pixels crossed by the line
        dx_pixels = abs(clipped_p2_2d[0] - clipped_p1_2d[0]) / self.pixel_size_x
        dy_pixels = abs(clipped_p2_2d[1] - clipped_p1_2d[1]) / self.pixel_size_y
        
        num_steps = int(max(dx_pixels, dy_pixels))
        if num_steps == 0:
            num_steps = 1 # Ensure at least one step for very short segments within a pixel

        # Calculate the length increment to add to each sampled pixel
        step_increment = segment_length_2d / num_steps

        # Iterate along the clipped 2D segment, distributing its length
        for i in range(num_steps + 1): # Include the end point for full coverage
            t = i / num_steps # Parameter along the segment from 0 to 1
            
            # Calculate the current point on the 2D segment
            current_point_x = clipped_p1_2d[0] + (clipped_p2_2d[0] - clipped_p1_2d[0]) * t
            current_point_y = clipped_p1_2d[1] + (clipped_p2_2d[1] - clipped_p1_2d[1]) * t

            # Convert local 2D coordinates to integer pixel indices
            col_idx = int(current_point_x / self.pixel_size_x)
            row_idx = int(current_point_y / self.pixel_size_y)

            # Ensure pixel indices are within the valid bounds of the pixel grid
            col_idx = min(max(0, col_idx), self.shape[0] - 1)
            row_idx = min(max(0, row_idx), self.shape[1] - 1)
            
            # Add the calculated length increment to the corresponding pixel
            self.pixels[row_idx, col_idx] += step_increment

    @property
    def get_pixels(self):
        """
        Returns the 2D PyTorch tensor representing the pixel grid with accumulated lengths.

        Returns:
            torch.Tensor: A 2D tensor of shape (num_pixels_width, num_pixels_length).
        """
        return self.pixels
