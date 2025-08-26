#!/usr/bin/env python
'''
A pedantic explanation of ray grid

A "ray" is conceptually an infinite line with a defined direction.  The ray line
is defined by giving two points on the line and the ray direction is taken as
going from the first point to the second point.  The points are specified in a
2D Cartesian coordinate system called "face" coordinates (not the same as but
related to WCT 3D coordinates).  This coordinate system takes the usual
convention of the x-axis pointing right on the horizontal and the y-axis
pointing up on the vertical.

A "ray grid" is an array of rays that are mutually parallel and uniformly spaced
by a fixed "pitch" distance measured perpendicularly between two neighboring
rays.  A "ray grid" is placed into the "face" coordinates by giving a "center"
point and a "pitch vector", both defined in the 2D "face" coordinates.  The
"center" point is on a special ray called "ray zero" and the "pitch vector" is
perpendicular to the rays and with magnitude equal to their transverse
separation.  The term "ray zero" is used as later rays are identified by indices
and ray zero is index 0.

Multiple ray grids of different angles may be overlayed in the "face"
coordinates.  Each is said to provide one tomographic "view" of a scalar field
defined in the "face" coordinate system.  An array of sensitive detectors are
associated with a finite subset of rays starting with "ray zero" in each view
spanning some active region.  Each detector measures the integral of the scalar
field in the vicinity of the ray and over the active region.  The array of
detectors produces an "activity" tensor, each element being one measured
integral.

In order to estimate the 2D scalar field given multiple views of 1D activity
measures, the algorithms provided wirecell.raygrid.tiling algorithm are applied.
They rely on the "Coordinates" class provided below for performing fast geometry
operation between views and between a particular view coordinate system with its
"ray" and "pitch" axes and the "face" coordinate system.  Regions along the
pitch axes in each view are identified by half-open ranges of "view indices".
The range [a,b) contains the ray at index "a" up to but excluding the ray at
index "b".  Rays are thus "lower edges" of half open bins.  Use of "view
indices" is the key to the optimizations in the Coordinate class and what will
be seen later in the wirecell.raygrid.tiling module.
'''

import torch

from . import funcs

class Coordinates: 

    # Attributes:

    # (Nview,) magnitude of pitch of each view
    pitch_mag=None
    # (Nview, 2) unit vector along the pitch direction of each view
    pitch_dir=None
    # (Nview, 2) origin vector of each view
    center=None
    # (Nview, Nview, 2) crossing point of a "ray zero" from a pair of views.
    # Undefined for views which are mutually parallel.
    zero_crossings=None
    # (Nview, Nview, 2) displacement vector along ray direction of the first
    # view between two crossing of that ray and two consecutive rays in the
    # second view.
    ray_jump=None
    # The ray grid "A" tensor.  See raygrid.pdf doc.
    a=None
    # The ray grid "B" tensor.  See raygrid.pdf doc.
    b=None

    # (Nview,2), Per-view unit vectors in direction of ray_jump.  Helper, not
    # needed for main ray-grid coordinate calculations
    ray_dir=None

    def __init__(self, views):
        '''
        Construct Ray Grid coordinates specified by views.

        The views is a 3-D tensor of shape:

            (N-views, 2 endpoints, 2 coordinates)

        Each view is a pair of endpoints.  The first point marks the origin of
        the view.  The relative vector from first to second point is in the
        direction of the pitch.  The magnitude of the vector is the pitch.
        '''
        self.init(views)
        self.views = views      # keep for provenance

    @property
    def nviews(self):
        return self.pitch_mag.shape[0]

    @property
    def bounding_box(self):
        '''
        Tensor of shape (2,2) holding [ (x0,x1), (y0,y1) ] bounds in Cartesian space.
        '''

        if self.pitch_dir[0,0] == 0: # points up, hbounds is view 0
            x0 = self.center[1,0]
            y0 = self.center[0,1]
            x1 = x0 + self.pitch_mag[1]
            y1 = y0 + self.pitch_mag[0]
        else:
            x0 = self.center[0,1]
            y0 = self.center[1,0]
            x1 = x0 + self.pitch_mag[0]
            y1 = y0 + self.pitch_mag[1]
        return torch.tensor([ [x0,x1], [y0,y1] ])


    def point_pitches(self, points: torch.Tensor) -> torch.Tensor:
        '''
        Return the pitch location measured in each view for a batch of 2D Cartesian points.

        Args:
            points: A tensor of shape (nbatch, 2) providing 2D Cartesian coordinates.

        Returns:
            A tensor of floating point values and shape (nbatch, nview)
            giving the per-view pitch for each point.
        '''
        if points.dim() != 2 or points.shape[1] != 2:
            raise ValueError("Input 'points' must be a tensor of shape (nbatch, 2).")
        
        # (Nview,)
        nviews = self.pitch_mag.shape[0] 
        # (nbatch, 2)
        nbatch = points.shape[0]

        # Reshape points to (nbatch, 1, 2) for broadcasting with view data
        points_reshaped = points.unsqueeze(1) # (nbatch, 1, 2)

        # Reshape center to (1, Nview, 2) for broadcasting with points
        center_reshaped = self.center.unsqueeze(0) # (1, Nview, 2)

        # Calculate vector from each view's center to each point
        # Result will be (nbatch, Nview, 2)
        # (nbatch, 1, 2) - (1, Nview, 2) -> (nbatch, Nview, 2)
        vec_center_to_point = points_reshaped - center_reshaped

        # Reshape pitch_dir to (1, Nview, 2) for broadcasting
        pitch_dir_reshaped = self.pitch_dir.unsqueeze(0) # (1, Nview, 2)

        # Calculate the dot product along the last dimension to get pitches.
        # This is a batched dot product: sum((nbatch, Nview, 2) * (1, Nview, 2)) over dim 2
        # Result will be (nbatch, Nview)
        pitches = torch.sum(vec_center_to_point * pitch_dir_reshaped, dim=2)

        return pitches

    def point_indices(self, points: torch.Tensor) -> torch.Tensor:
        '''
        Return the integer pitch index in each view for a batch of 2D Cartesian points.

        Args:
            points: A tensor of shape (nbatch, 2) providing 2D Cartesian coordinates.

        Returns:
            A tensor of integer values and shape (nbatch, nviews)
            giving the pitch index in each view for each point.
        '''
        # Calculate the floating-point pitch locations for each point in each view
        # (nbatch, nview)
        pitches_per_view = self.point_pitches(points)

        # Get nviews from the calculated pitches_per_view (or self.pitch_mag)
        nviews = pitches_per_view.shape[1]

        # Reshape pitch_mag to (1, Nview) for broadcasting with pitches_per_view
        # (nbatch, Nview) / (1, Nview) -> (nbatch, Nview)
        # Note: self.pitch_mag is (Nview,)
        pitch_mag_reshaped = self.pitch_mag.unsqueeze(0)

        # Calculate pitch indices using the existing pitch_index logic: floor(pitch / pitch_mag)
        # The original pitch_index method takes pitch (scalar/1D) and view (scalar/1D)
        # but here we are effectively applying it across all views for all points.
        # We can directly perform the division and floor operation here,
        # as pitch_index just wraps that.
        
        # (nbatch, nview)
        indices_float = pitches_per_view / pitch_mag_reshaped

        # Apply floor and convert to long integer type
        # (nbatch, nview)
        pitch_indices = torch.floor(indices_float).to(torch.long)

        return pitch_indices

    def ray_crossing(self, view1, ray1, view2, ray2):
        '''
        Return the 2D crossing point(s) ray grid coordinates "one" and
        "two".  Each coordinate is given as a pair (view,ray) of indices.  These
        may be scalar or batched array.
        '''
        r00 = self.zero_crossings[view1, view2]
        w12 = self.ray_jump[view1, view2]
        w21 = self.ray_jump[view2, view1]

        # broadcast matching
        ray1 = ray1.unsqueeze(1)
        ray2 = ray2.unsqueeze(1)

        return r00 + ray2 * w12 + ray1 * w21;

    def pitch_location(self, view1, ray1, view2, ray2, view3):
        '''
        Return the pitch location measured in the given view (an index) of
        the crossing point of ray grid coordinates one and two.
        '''
        return self.b[view1, view2, view3] \
            + ray2 * self.a[view1, view2, view3] \
            + ray1 * self.a[view2, view1, view3]


    def pitch_index(self, pitch, view):
        '''
        Return the index of the closest ray at a location in the view that
        is less than or equal to the given pitch.
        '''
        return torch.floor(pitch/self.pitch_mag[view]).to(torch.long)


    def init(self, pitches):
        '''
        Initialize or reinitialize the coordinate system  
        '''
        
        nviews = pitches.shape[0]

        # 1D (l) the magnitude of the pitch of view l.
        pvrel = pitches[:,1,:] - pitches[:,0,:]
        self.pitch_mag = torch.sqrt(pvrel[:,0]**2 + pvrel[:,1]**2)

        # 2D (l,c) the pitch direction 2D coordinates c of view l.
        self.pitch_dir = pvrel / self.pitch_mag.reshape(nviews,1)

        # 2D (l,c) the 2D coordinates c of the origin point of view l
        self.center = pitches[:,0,:]

        self.ray_dir = torch.vstack((-self.pitch_dir[:,1], self.pitch_dir[:,0])).T
        ray0 = torch.vstack((self.center - self.ray_dir, self.center + self.ray_dir)).reshape(2,-1,2)
        ray1 = torch.vstack((ray0[0] + pvrel, ray0[1] + pvrel)).reshape(2,-1,2)


        # 3D (l,m,c) crossing point 2D coordinates c of "ray 0" of views l and
        # m.  
        self.zero_crossings = torch.zeros((nviews, nviews, 2))

        # 3D (l,m,c) difference vector coordinates c between two consecutive
        # m-view crossings along l ray direction.  between crossings of rays of
        # view m.  
        self.ray_jump = torch.zeros((nviews, nviews, 2))

        # The Ray Grid tensor representations.
        self.a = torch.zeros((nviews, nviews, nviews))
        self.b = torch.zeros((nviews, nviews, nviews))        

        # Cross-view things
        for il in range(nviews):
            rl0 = ray0[:,il,:]
            rl1 = ray1[:,il,:]

            for im in range(nviews):
                rm0 = ray0[:,im,:]
                rm1 = ray1[:,im,:]

                # Special case diagonal values
                if il == im:
                    # This is redundant in some sense but conceptually recasts
                    # the idea of a view origin as also the ray-zero crossing
                    # with itself.
                    self.zero_crossings[il,im] = self.center[il]
                    # Likewise, one can not "jump" along a view direction
                    # between two ray crossings of the view with itself.  But,
                    # the ray jump direction does characterize the "self jump".
                    self.ray_jump[il,im] = funcs.ray_direction(rl0)
                    continue;
                

                if il < im:
                    # Fill in both triangles in one go to exploit the symmetry of this:
                    try:
                        p = funcs.crossing(rl0, rm0)
                    except ValueError:
                        print(f'skipping parallel view pair: {il=} {im=}')
                        continue

                    self.zero_crossings[il, im] = p
                    self.zero_crossings[im, il] = p
                    self.ray_jump[il, im] = funcs.crossing(rl0, rm1) - p
                    self.ray_jump[im, il] = funcs.crossing(rm0, rl1) - p

        # Triple layer things
        for ik, pk in enumerate(self.pitch_dir):
            cp = torch.dot(self.center[ik], pk)

            for il in range(nviews):
                if il == ik:
                    continue

                for im in range(il):
                    if im == ik:
                        continue

                    rlmpk = torch.dot(self.zero_crossings[il, im], pk)
                    wlmpk = torch.dot(self.ray_jump[il, im], pk)
                    wmlpk = torch.dot(self.ray_jump[im, il], pk)
                    self.a[il,im,ik] = wlmpk
                    self.a[im,il,ik] = wmlpk;
                    self.b[il,im,ik] = rlmpk - cp
                    self.b[im,il,ik] = rlmpk - cp
            
    def __str__(self):
        return self.as_string()

    def as_dict(self):
        return dict(
            views = self.views,
            pitch_mag = self.pitch_mag,
            pitch_dir = self.pitch_dir,
            center = self.center,
            zero_crossings = self.zero_crossings,
            ray_jump = self.ray_jump,
            ray_dir = self.ray_dir,
            a = self.a,
            b = self.b)

    def as_string(self, style=None):
        if style in (None, "python"):
            return str(self.as_dict())
        if style.lower() in ("c++"):
            lines = ["const std::map<std::string, torch::Tensor> ray_grid_coordinates_data = {"]
            for k,v in self.as_dict().items():
                v = v.to(torch.double)
                t = str(v).replace("[","{").replace("]","}")
                t = t.replace("dtype=torch.float64", "torch::kDouble")
                lines += [f'    {{ "{k}", torch::{t} }},']
            lines[-1] = lines[-1][:-1] # remove final comma
            lines.append('};')
            return '\n'.join(lines)

