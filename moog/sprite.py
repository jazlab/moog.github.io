# This file was forked and heavily modified from the file here:
# https://github.com/deepmind/spriteworld/blob/master/spriteworld/sprite.py
# Here is the license header for that file:

# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Sprite.

The main class in this file is Sprite. Instances of Sprite comprise the state of
an environment. This file also contains some helper functions for updating the 
attributes of a sprite and computing crossing points of lines and sprite edges.
"""

import collections
from matplotlib import path as mpl_path
from matplotlib import transforms as mpl_transforms
import numpy as np
from moog import shapes

GLOBAL_SPRITE_COUNT = 0

# Tiny float for numerical stability in segment_crossings()
_EPSILON_INTERPOLATION = 1e-8


def update_sprite(sprite, **factors):
    """Update sprite in place given an entirely new set of factors.

    This file updates a sprite as efficienctly as possible, i.e. without
    resetting the shape or transforming the path unless those are necessary.

    Args:
        sprite: Instance of Sprite.
        **factors: Dict. Keys must be in Sprite.FACTOR_NAMES. Values are new
            values for those factors.
    """
    # Some factors can be set directly
    for k in ['c0', 'c1', 'c2', 'opacity', 'angle_vel', 'mass', 'metadata']:
        if k in factors:
            setattr(sprite, k, factors[k])

    # Set position
    if 'x' in factors or 'y' in factors:
        sprite.position = [
            factors.get('x', sprite.position[0]),
            factors.get('y', sprite.position[1]),
        ]

    # Set velocity
    if 'x_vel' in factors or 'y_vel' in factors:
        sprite.velocity = [
            factors.get('x_vel', sprite.velocity[0]),
            factors.get('y_vel', sprite.velocity[1]),
        ]

    # Setting shape, angle, scale, or aspect_ratio requires transforming the
    # sprite path, which is somewhat computationally expensive, so for these we
    # check to see if they've changed before setting them.
    for k in ['angle', 'scale', 'aspect_ratio']:
        if k in factors and getattr(sprite, k) != factors[k]:
            setattr(sprite, k, factors[k])
    if 'shape' in factors:
        if isinstance(factors['shape'], str):
            # Shape is encoded as a string, not as vertices, so we can just
            # reset the shape attribute
            if sprite.shape != factors['shape']:
                sprite.shape = factors['shape']
        else:
            # Shape is encoded as an array of vertices.
            if not np.array_equal(sprite.vertices, factors['shape']):
                # Note that we could invert the angle, scale, aspect_ratio, and
                # position transformations to then use the sprite's shape
                # setter, but this would add about 15 lines of code and would
                # not be very readable. So instead we take a shortcut and
                # directly override the protected sprite._path attribute.
                vertices = np.concatenate(
                    (factors['shape'], factors['shape'][:1]), axis=0)
                sprite._path = mpl_path.Path(vertices)  #pylint: disable=protected-access

    return


def segment_crossing_coefficients(start_0, end_0, start_1, end_1):
    """Solves linear equations to compute crossing points of segments.

    Finding the crossing point between segment [start_0, end_0] and segment
    [start_1, end_1] can be done by solving
        start_0 + A * delta_0 = start_1 + B * delta_1
    for A and B, where delta_i = end_i - start_i.

    The solution (by a little vector algebra) is
        A = ((start_1 - start_0) x delta_1) / (delta_0 x delta_1)
        B = ((start_1 - start_0) x delta_0) / (delta_0 x delta_1)
    where x denotes cross product. If A and B are both in [0, 1], then the
    segments [start_0, end_0] and [start_1, end_1] intersect at point
    start_0 + A * delta_0.

    This function does this computation vectorized to handle multiple segments
    at once, returning the coefficients A and B. Namely, it computes all
    pairwise crossings between two arrays of segments.

    The reason this function is factored out of segment_crossings() below is
    because some calling code (e.g. collisions) need to use the A and B
    coefficients.

    Args:
        start_0: Numpy array of shape [N, 2]. Starting points for set 0.
        end_0: Numpy array of shape [N, 2]. Ending points for set 0.
        start_1: Numpy array of shape [M, 2]. Starting points for set 1.
        end_1: Numpy array of shape [M, 2]. Ending points for set 1.

    Returns:
        A: Numpy array of shape [N, N]. A[i, j] is the coefficient A (see above)
            for the crossing of segment [start_0[i], end_0[i]] and the segment
            [start_1[j], end_1[j]].
        B: Numpy array of shape [N, N]. B[i, j] is the coefficient B (see above)
            for the crossing of segment [start_0[i], end_0[i]] and the segment
            [start_1[j], end_1[j]].
    """
    s_0 = start_0
    ds_0 = end_0 - start_0
    s_1 = start_1
    ds_1 = end_1 - start_1

    # Expand the _0's and _1's in different dimensions to leverage broadcasting.
    s_0 = s_0[:, np.newaxis]
    ds_0 = ds_0[:, np.newaxis]
    s_1 = s_1[np.newaxis]
    ds_1 = ds_1[np.newaxis]

    # Need to add small epsilon for stability, else may divide by zero below
    ds_0_cross_ds_1 = np.cross(ds_0, ds_1) + _EPSILON_INTERPOLATION
    s_1_minus_s_0 = s_1 - s_0

    A = np.cross(s_1_minus_s_0, ds_1) / ds_0_cross_ds_1
    B = np.cross(s_1_minus_s_0, ds_0) / ds_0_cross_ds_1

    return A, B


def segment_crossings(start_0, end_0, start_1, end_1):
    """Finds all pairwise crossings between two arrays of segments.

    See segment_crossing_coefficients() documentation above for an explanation
    of how this is done. In contrast to segment_crossing_coefficients(), this
    function outputs the actual crossing points themselves.

    Args:
        start_0: Numpy array of shape [N, 2]. Starting points for set 0.
        end_0: Numpy array of shape [N, 2]. Ending points for set 0.
        start_1: Numpy array of shape [M, 2]. Starting points for set 1.
        end_1: Numpy array of shape [M, 2]. Ending points for set 1.

    Returns:
        crossing_points: Numpy array of shape [K, 2] containing all crossing
            points. Here K is the number of crossing points.
        inds_crossings: Numpy array of shape [K, 2] containing indices for
            segments_0 and segments_1 respectively. Specifically,
            inds_crossings[i][0] is the index in [0, N] of the set 0 segment and
            inds_crossings[i][1] is the index in [0, M] of the set 1 segment in
            crossing point i.
    """
    A, B = segment_crossing_coefficients(start_0, end_0, start_1, end_1)

    # Crossings occur when A and B are both in [0, 1]
    crossings = (A > 0) * (A < 1) * (B > 0) * (B < 1)
    inds_crossings = np.argwhere(crossings)

    # Linear combination of segment 0 with A coefficients gives crossing points
    crossing_points = np.array([
        start_0[ind_0] + A[ind_0, ind_1] * (end_0[ind_0] - start_0[ind_0])
        for (ind_0, ind_1) in inds_crossings
    ])
    return crossing_points, inds_crossings


def sprite_edge_crossings(sprite_0, sprite_1):
    """Find all points where the boundary of sprite_0 and sprite_1 cross.

    See segment_crossings() documentation for details.

    Args:
        sprite_0: Instance of Sprite.
        sprite_1: Instance of Sprite.

    Returns:
        crossing_points: Numpy float array of shape [K, 2] where K is the number
            of crossings of the boundaries of sprite_0 and sprite_1.
        inds_crossings: Numpy ind array of shape [K, 2].
    """
    path_0 = sprite_0.path.vertices
    path_1 = sprite_1.path.vertices

    crossing_points, inds_crossings = segment_crossings(
        start_0=path_0[:-1],
        end_0=path_0[1:],
        start_1=path_1[:-1],
        end_1=path_1[1:],
    )

    return crossing_points, inds_crossings


class Sprite(object):
    """Sprite class.

    Sprites are polygons parameterized by a few factors (such as position,
    shape, angle, scale, color, velocity, and mass). They are the building
    blocks of the environment and the objects upon which physics can act.
    """

    FACTOR_NAMES = (
        'x',  # x-position of sprite center-of-mass (float)
        'y',  # y-position of sprite center-of-mass (float)
        'shape',  # shape (string)
        'angle',  # angle in radians (float)
        'scale',  # size of sprite (float)
        'aspect_ratio',  # aspect ratio of sprite (float)
        'c0',  # first color component (scalar)
        'c1',  # second color component (scalar)
        'c2',  # third color component (scalar)
        'opacity',  # opacity of sprite in [0, 255]
        'x_vel',  # x-component of velocity (float)
        'y_vel',  # y-component of velocity (float)
        'angle_vel',  # angular velocity in radians/time
        'mass',  # mass
        'metadata',  # optional metadata
    )

    # Shape factor name for shapes not in shapes.SHAPES
    _CUSTOM_SHAPE = 'custom'

    # Name for a circle. Must match that in shapes.SHAPES
    _CIRCLE_NAME = 'circle'

    def __init__(self,
                 x=0.5,
                 y=0.5,
                 shape='square',
                 angle=0.,
                 scale=1.,
                 aspect_ratio=1.,
                 c0=0,
                 c1=0,
                 c2=0,
                 opacity=255,
                 x_vel=0.,
                 y_vel=0.,
                 angle_vel=0.,
                 mass=1.,
                 metadata=None):
        """Construct sprite.

        This class is agnostic to the color scheme, namely (c1, c2, c3) could be
        in RGB coordinates or HSV, HSL, etc. without this class knowing. The
        color scheme conversion for rendering must be done in the renderer.

        Args:
            x: Float in [0, 1]. x-position.
            y: Float in [0, 1]. y-position.
            shape: String or numpy array of vertices. Shape of the sprite. If
                string, must be a key of shapes.SHAPES. If array, must have
                shape [N, 2] defining the vertices of the polygonal shape.
            angle: Float. Angle in radians.
            scale: Float. Scale (size) of the sprite. This is multiplied to the
                shape vertex array when constructing the sprite, hence the
                sprite width scales linearly with respect to scape, and sprite
                area scales with power 2.
            aspect_ratio: Scalar. Height/width aspect ratio.
            c0: Scalar. First coordinate of color.
            c1: Scalar. Second coordinate of color.
            c2: Scalar. Third coordinate of color.
            opacity: Integer in [0, 255]. Opacity of sprite.
            x_vel: Float. x-velocity.
            y_vel: Float. y-velocity.
            angle_vel: Float. Angular velocity in radians/time.
            mass: Float. Mass.
            metadata: Any type. Optional metadata. If None, defaults to empty
                dictionary.
        """
        # The angle, scale, and aspect ratio must be converted to floats.
        # Sometimes the state initializer distribution will feed them in as
        # numpy scalars but this causes the updating to not work.
        self._position = np.array([x, y])
        self._angle = float(angle)
        self._scale = float(scale)
        self._aspect_ratio = float(aspect_ratio)
        self._color = (c0, c1, c2)
        self._opacity = opacity
        self._velocity = np.array([x_vel, y_vel])
        self._angle_vel = angle_vel
        self._mass = mass
        self.metadata = metadata

        # This calls shape.setter, which does shape path setting
        self.shape = shape

        # Increment the global sprite count and set self._id, which can be used
        # to identify sprite instances.
        global GLOBAL_SPRITE_COUNT
        self._id = GLOBAL_SPRITE_COUNT
        GLOBAL_SPRITE_COUNT += 1

    def _set_shape_path(self, shape_path):
        """Set shape path and moment of inertia.

        The self._shape_path set by this function is centered around the shape's
        center of mass, assuming a uniform mass distribution. Moment of inertia
        is calculated in tandem, to avoid having to iterate through vertices
        multiple times.

        The rotational moment of intertia of a 2-dimensional body about the
        z-axis (i.e. perpendicular to the plane and passing through the origin):
            I = density * integral_{area}[x^2 + y^2 dx dy]
              = I_x + I_y

        This is important to have because it is needed to relate torque to
        rotational acceleration via Newton's Law:
            torque = I * rotational_acceleration
        This is needed in our environment to simulate realistic physical
        collisions when sprites are allowed to rotate.

        In this function we compute I_x and I_y themselves and set
            self._x_y_rotational_inertia = [I_x, I_y]
        The reason to have these components is to allow them to be adjusted upon
        changes in aspect ratio without having to re-compute them from scratch
        by re-running this function. See self._set_path().

        Also see self.moment_of_inertia for the summed moment of inertia I.

        Computing I_x and I_y of a polygon can be done by breaking the polygon
        into triangles. Then for a triangle with one vertex at the origin the
        above integral can be solved analytically.
        """
        # Compute centroid, intertia, and area
        num_vertices = shape_path.shape[0]
        inertia = np.array([0., 0.])
        area = 0.
        centroid = np.array([0., 0.])
        for i in range(num_vertices):
            vertex_0 = shape_path[i]
            vertex_1 = shape_path[(i + 1) % num_vertices]

            # x and y moments of inertia
            cross = np.cross(vertex_0, vertex_1)
            inertia += (1. / 12.) * cross * (
                vertex_0 * vertex_0 + vertex_1 * vertex_1 + vertex_0 * vertex_1)

            # area and centroid
            triangle_area = cross / 2.
            triangle_centroid = (vertex_0 + vertex_1) / 3.
            area += triangle_area
            centroid += triangle_centroid * triangle_area
        centroid /= area

        if area < 0:
            # Path was defined clockwise, which will mess up collisions, so we
            # reverse the path and correct for the negative area and inertia.
            # The centroid is fine, because was divided by the area.
            shape_path = shape_path[::-1]
            inertia *= -1.
            area *= -1.

        # Set shape path with center of mass at origin
        center_translate = mpl_transforms.Affine2D().translate(*(-1 * centroid))
        # Make shape path be a full loop + 1 (i.e. last point in the array is
        # the first point). This makes calculating sprite overlaps easier,
        # because mpl_path.Path.intersects_path requires the full loop + 1.
        shape_path = np.concatenate((shape_path, [shape_path[0]]))
        self._shape_path = center_translate.transform_path(
            mpl_path.Path(shape_path))

        # Use parallel axis theorem to compute moment of inertia around center
        # of mass, so we don't have to iterate through the points again
        inertia -= area * np.square(centroid)
        self._x_y_rotational_inertia = inertia / area

        # We must call self._set_path() before updating self.position, becuase
        # the position setter uses self._path, which is set in self._set_path()
        self._set_path()
        self.position = self._position + centroid

        # Make sure self._just_set_shape is True
        self._just_set_shape = True

    def _set_path(self):
        """Rotate and scale self._shape path."""
        x_y_scale = np.array([self._scale, self._scale * self._aspect_ratio])
        transform = (
            mpl_transforms.Affine2D().scale(*x_y_scale) +
            mpl_transforms.Affine2D().rotate(self._angle) +
            mpl_transforms.Affine2D().translate(*self._position))
        self._path = transform.transform_path(self._shape_path)
        self._max_radius = np.max(
            np.linalg.norm(self.vertices - self._position, axis=1))

        # Adjust rotational inertia to accomodate the change in aspect ratio and
        # scale
        self._x_y_rotational_inertia *= np.square(x_y_scale)

    def update_pos_from_vel(self, delta_t):
        """Update position based on velocity."""
        self.position = self.position + delta_t * self.velocity
        if self._angle_vel:
            self.angle = self.angle + delta_t * self._angle_vel

    def contains_point(self, point):
        """Check if the point is contained in the Sprite."""
        if self.is_symmetric_circle:
            contains_point = (
                np.linalg.norm(point - self.position) < self.max_radius)
        else:
            contains_point = self.path.contains_point(point)

        return contains_point

    def contains_points(self, points):
        """Check if the points are contained in the Sprite.

        Args:
            points: Numpy array of size (N, 2) containing coordinates of N
                points.

        Returns:
            contains_points: Boolean numpy array of size (N,), indicating for
                each point whether it is in this sprite.
        """
        if self.is_symmetric_circle:
            contains_points = (
                np.linalg.norm(points - self.position, axis=1) <=
                self.max_radius)
        else:
            contains_points = self._path.contains_points(points)

        return contains_points

    def overlaps_sprite(self, sprite):
        """Check if this and the argument sprite overlap."""
        center_dist = np.linalg.norm(self.position - sprite.position)
        if center_dist > self.max_radius + sprite.max_radius:
            return False

        # WARNING: You might think that the case of two circles depends only on
        # their radius, but using this shortcut causes instability in
        # collisions, since there is currently no special treatment for circles
        # in collisions (collisions depend on the vertices themselves). So we
        # give no special treatment to circles here.

        # Note: mpl_path.Path.intersects_path() treats its input paths as
        # paths with endpoints. Namely, an input array of N vertices
        # corresponds to a path with N - 1 edges. Thus to apply it to detect
        # intersection of sprites, we must feed it an array of length
        # len(self.vertices) + 1. In fact self._path is an array of length
        # len(self.vertices) + 1, by construction (see self._set_path()).
        # Profiling note: Using mpl_path.Path.intersects_path() is slightly
        # faster than `len(sprite_edge_crossings(self, sprite)[1] > 0)`.
        overlap = mpl_path.Path.intersects_path(
            self.path, sprite.path, filled=True)
        return overlap

    @property
    def vertices(self):
        """Numpy array of vertices of the shape."""
        return self.path.vertices[:-1]

    @property
    def path(self):
        """Numpy array of length len(self.vertices) + 1, loop of the shape."""
        return self._path

    @property
    def max_radius(self):
        return self._max_radius

    @property
    def is_symmetric_circle(self):
        return self.shape == Sprite._CIRCLE_NAME and self.aspect_ratio == 1

    @property
    def x(self):
        return self._position[0]

    @property
    def y(self):
        return self._position[1]

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        if isinstance(shape, str) and shape in shapes.SHAPES:
            self._shape = shape
            shape_path = shapes.SHAPES[shape]
        else:
            self._shape = Sprite._CUSTOM_SHAPE
            shape_path = shape

        self._set_shape_path(shape_path)

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, a):
        if id(a) == id(self._angle):
            # See comment in @position.setter for why we catch this.
            raise ValueError(
                'Cannot call in-place operations on sprite.angle.')
        rotate = mpl_transforms.Affine2D().rotate_around(
            self.x, self.y, a - self._angle)
        self._path = rotate.transform_path(self._path)
        self._angle = a

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, s):
        self._scale = s
        self._set_path()

    @property
    def aspect_ratio(self):
        return self._aspect_ratio

    @aspect_ratio.setter
    def aspect_ratio(self, a):
        self._aspect_ratio = a
        self._set_path()

    @property
    def c0(self):
        return self._color[0]

    @c0.setter
    def c0(self, c0):
        self._color = (c0, self.c1, self.c2)

    @property
    def c1(self):
        return self._color[1]

    @c1.setter
    def c1(self, c1):
        self._color = (self.c0, c1, self.c2)

    @property
    def c2(self):
        return self._color[2]

    @c2.setter
    def c2(self, c2):
        self._color = (self.c0, self.c1, c2)

    @property
    def opacity(self):
        return self._opacity

    @opacity.setter
    def opacity(self, opacity):
        self._opacity = opacity

    @property
    def x_vel(self):
        return self._velocity[0]

    @property
    def y_vel(self):
        return self._velocity[1]

    @property
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, mass):
        self._mass = mass

    @property
    def color(self):
        return self._color

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, pos):
        if id(pos) == id(self.position):
            # This setter references the pre-set position self._position, but
            # with an in-place operation, the pre-set position is the same array
            # as `pos`, the position to be set, so this setter function does not
            # work as intended. In practice, this means that an in-place
            # operation updates self._position but doesn't update self._path,
            # which causes sprites to not move. Consequently, calling code must
            # not update position in place. That can cause devious bugs if
            # undetected, so we catch it with this ValueError.
            raise ValueError(
                'Cannot call in-place operations on sprite.position.')
        if not isinstance(pos, np.ndarray):
            pos = np.array(pos)
        translate = mpl_transforms.Affine2D().translate(*pos - self._position)
        self._path = translate.transform_path(self._path)
        self._position = pos

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, vel):
        if not isinstance(vel, np.ndarray):
            vel = np.array(vel)
        self._velocity = vel

    @property
    def angle_vel(self):
        return self._angle_vel

    @angle_vel.setter
    def angle_vel(self, angle_vel):
        self._angle_vel = angle_vel

    @property
    def just_set_shape(self):
        """This property can used and set by loggers."""
        return self._just_set_shape

    @just_set_shape.setter
    def just_set_shape(self, just_set_shape):
        self._just_set_shape = just_set_shape

    @property
    def moment_of_inertia(self):
        return sum(self.mass * self._x_y_rotational_inertia)

    @property
    def id(self):
        return self._id

    @property
    def factors(self):
        factors = collections.OrderedDict()
        for factor_name in Sprite.FACTOR_NAMES:
            factors[factor_name] = getattr(self, factor_name)
        return factors
