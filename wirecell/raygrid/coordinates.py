#!/usr/bin/env python
'''
Ray grid coordinates provide a set of 1D tomographic coordinate systems
("views") that cover a 2D domain.

The 1D coordinates of each view as a unique sampling period ("pitch magnitude")
and direction ("pitch direction").  The start of each sample is a line ("ray")
that is perpendicular to the pitch.  One coordinate represents the half-open 2D
region ("strip" 0bounded by the ray and the next neighboring ray at higher
pitch.

The shared angle of each view's set of parallel rays is unique among the views.
Each view may have a unique pitch magnitude.  Rays are identified by
non-negative indices and a point on "ray 0" is identified as the "view origin".

Each view is likewise identified with a non-negative "layer index".  The views
are uniquely specified with an ordered list of pairs of line segments.  Each
pair is coincident with the view's "ray 0" and "ray 1" rays and such that the
first segment is centered on the view's origin.
'''

import torch

from . import funcs

class Coordinates: 

    def __init__(self, views):
        '''
        Construct Ray Grid coordinates specified by views.

        The views is a 4-D tensor of shape:

            (N-views, 2 rays, 2 endpoints, 2 coordinates)

        '''
        self.init(views)
    
    def ray_crossing(self, one, two):
        '''
        Return the 2D crossing point(s) ray grid coordinates "one" and
        "two".  Each coordinate is given as a pair (view,ray) of indices.  These
        may be scalar or batched array.
        '''
        view1, ray1 = one
        view2, ray2 = two

        r00 = self.zero_crossings[view1, view2]
        w12 = self.ray_jump[view1, view2]
        w21 = self.ray_jump[view2, view1]
        return r00 + ray2 * w12 + ray1 * w21;

    def pitch_location(self, one, two, view):
        '''
        Return the pitch location measured in the given view (an index) of
        the crossing point of ray grid coordinates one and two.
        '''
        view1, ray1 = one
        view2, ray2 = two
        
        return self.b[view1, view2, view] \
            + ray2 * self.a[view1, view2, view] \
            + ray1 * self.a[view2, view1, view]


    def init(self, views):
        '''
        Initialize or reinitialize the coordinate system  
        '''
        
        nviews = views.shape[0]
        # 1D (l) the magnitude of the pitch of view l.
        self.pitch_mag = torch.zeros((nviews,))
        # 2D (l,c) the pitch direction 2D coordinates c of view l.
        self.pitch_dir = torch.zeros((nviews, 2))
        # 2D (l,c) the 2D coordinates c of the origin point of view l
        self.center = torch.zeros((nviews, 2))
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

        # Per-view things
        for layer, (r0, r1) in enumerate(views):
            rpv = funcs.pitch(r0, r1)

            rpl = torch.norm(rpv)
            self.pitch_mag[layer] = rpl
            self.pitch_dir[layer] = rpv/rpl
            self.center[layer] = 0.5 * (r0[0] + r0[1])

        # Cross-view things
        for il, (rl0, rl1) in enumerate(views):
            for im, (rm0, rm1) in enumerate(views):            

                # Special case diagonal values
                if il == im:
                    # This is redundant in some sense but conceptually recasts
                    # the idea of a view origin as also the ray-zero crossing
                    # with itself.
                    self.zero_crossings[il,im] = self.center[il]
                    # Likewise, one can not "jump" along a view direction
                    # between two ray crossings of the view with itself.  But,
                    # the ray jump direction does characterize the "self jump".
                    self.ray_jump[il,im] = funcs.direction(rl0)
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
            
