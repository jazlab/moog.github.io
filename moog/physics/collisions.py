"""Collision forces.

The main class in this file is Collision. It simulates Newtonian collisions
between two sprites. These collisions take into account the angles of the sprite
edges/vertices at the point of contact to realistically simulate the collision.
By default they respect Newtonian mechanics for rotation as well, taking into
consideration the moments of inertia of the sprites. However, there is an option
to ignore this and prohibit the collision from changing the angular velocity of
the sprites.

WARNING: Sprites must be star-shaped with respect to their center of mass (i.e.
the line fron the center of mass to any point in the sprite never exits the
sprite) for guaranteed correct collisions. If they are not star-shaped, there
can be rare events where two sprites collide at a corner and the collision is
not simulated correctly.
"""

# TODO(nwatters): Make collisions more robust when multiple collisions happen to
# a sprite in a single timestep. Consider using a backend like PyBullet or
# PyMunk.

# TODO(nwatters): Consider giving special treatment to circles for the collision
# point detection functions. Circular sprites are commonly used and have many
# vertices, which makes the _get_collision_vectors() function slow. Inferring
# collision point based on radius for circles might be worth making the code a
# little messier.

# TODO(nwatters): Handle the case where one sprite is dropped inside of another
# without having moved there by its physics. This can occur when the sprite is
# subject to some dynamics (e.g. via an action space or a portal) that directly
# change its position (instead of just changing its velocity). In this case the
# sprite should pop out of the overlap region. The current code does not handle
# this, but it is an important feature and would entail changing the
# _directed_collision_vectors() function.

from . import abstract_force
from matplotlib import path as mpl_path
from matplotlib import transforms as mpl_transforms
import numpy as np
from moog import sprite as sprite_lib

# Small float to ensure sprite position correction is conservative. Make sure
# this is not too small though, or ._make_disjoint() will be called
# unnecessarily often in Collision.step(). Something around 1e-2 seems to be
# fine.
_EPSILON = 1e-2


def _relative_motion_trajectory(path, path_sprite, anchor_sprite, delta_t):
    """Find trajectory of a path in the coordinate frame of an anchor sprite.

    Here path is rigidly tethered to path_sprite, i.e. it may consist of
    vertices of path_sprite. This function produces the trajectory of the path
    points in the inertial frame of anchor_sprite, i.e. the moving coordinate
    system in which anchor_sprite is static.

    Args:
        path: Instance of mpl_path.Path. path.vertices has shape [N, 2].
        path_sprite: Instance of ../sprite.Sprite.
        anchor_sprite: Instance of ../sprite.Sprite.
        delta_t: Scalar. Timestep over which to produce the trajectories.

    Returns:
        trajectory: Numpy array of shape [N, 2, 2]. Element i of trajectory
            is an array of shape [2, 2] in which the second element is the i'th
            path vertex and the first element is the previous position of that
            vertex in the relative coordinate transformation between
            anchor_sprite and path_sprite.
    """
    transform = (
        mpl_transforms.Affine2D().rotate_around(
            *path_sprite.position,
            theta=-1 * path_sprite.angle_vel * delta_t) +
        mpl_transforms.Affine2D().translate(
            *(-1 * path_sprite.velocity * delta_t)) +
        mpl_transforms.Affine2D().rotate_around(
            *anchor_sprite.position,
            theta=anchor_sprite.angle_vel * delta_t) +
        mpl_transforms.Affine2D().translate(
            *anchor_sprite.velocity * delta_t)
    )
    previous_path = transform.transform_path(path)
    trajectory = np.stack((previous_path.vertices, path.vertices), axis=1)

    return trajectory


def _directed_collision_vectors(sprite_0, sprite_1, delta_t):
    """Find single contact point and collision vectors.
    
    This function does the following:
        1. Finds all vertices of sprite_0 that lie inside of sprite_1.
        2. Finds the true continuous-time point of contact for each of these.
        3. Selects on that is most likely to be the true collisions point.
        4. Produces that collision point, the collision normal vector, and
            relative displacement of the sprites since the true collision time.

    Note that this is directed, i.e. only considers vertices of sprite_0 that
    lie in sprite_1. In reality a collision might occur by a vertex of sprite_1
    initially entering sprite_0. Hence we must call this function on
    *(sprite_1, sprite_0) as well. See _get_collision_vectors() for details.

    Args:
        sprite_0: Instance of ../sprite.Sprite.
        sprite_1: Instance of ../sprite.Sprite.
        delta_t: Scalar. Timestep of the physics.

    Returns:
        collision_point: None or numpy array of shape [2]. Position of the
            collision.
        collision_normal: None or numpy array of shape [2]. Zero-centered vector
            indicating the normal of the collision surface, i.e. the direction
            in which the collision force is applied. It is signed, so we let it
            point toward the sprite_1 side of the collision.
        since_collision: None or numpy array of shape [2]. Relative displacement
            of the two sprites since the collision. It is signed, so we let it
            point toward the sprite_0 side of the collision, i.e. representing
            how much sprite_0 has moved relative to sprite_0 since the collision
            event.
        perpendicular: A scaling of collision_normal, where the length is the
            margin of overlap of the sprites. This can be used to make the
            sprites disjoint.
    """
    # First find points of sprite_0 inside sprite_1
    vertices_0 = sprite_0.vertices
    contained_inds = np.argwhere(sprite_1.contains_points(vertices_0))[:, 0]
    if len(contained_inds) == 0:
        return None, None, None, None
    contained_vertices_0 = vertices_0[contained_inds]

    # Now find previous-to-current timestep trajectory of each contained vertex
    traj = _relative_motion_trajectory(
        mpl_path.Path(contained_vertices_0),
        sprite_0,
        sprite_1,
        delta_t,
    )

    # We will now find the crossing points between the previous-to-current
    # timestep relative trajectories of the vertices of sprite 0 and the edges 
    # of sprite 1. Instead of restricting ourselves to only the crossings of
    # those segments, we also extrapolate the relative trajectory of sprite 0
    # backwards in time to consider crossing points which may have occured
    # multiple timesteps ago. This is important to do because the relative
    # velocity of the sprites may have changed recently (e.g. due to some force
    # or action), so considering multi-timestep historical crossings avoids
    # instabilities. We can easily compute these historical crossings from the
    # segment crossing coefficients.
    vertices_1 = sprite_1.vertices
    vertices_1_next = np.concatenate((vertices_1[1:], vertices_1[:1]), axis=0)
    cross_a, cross_b = sprite_lib.segment_crossing_coefficients(
        start_0=traj[:, 0],
        end_0=traj[:, 1],
        start_1=vertices_1,
        end_1=vertices_1_next,
    )

    # To find current segment crossings we would further require cross_a > 0 and
    # cross_a < 1, but by allowing cross_a to be negative we let in historical
    # crossings and by allowing cross_a to be greater than 1 we let in future
    # crossings.
    crossings = (cross_b >= 0) * (cross_b <= 1)

    if not np.any(crossings):
        # This can happen because of _EPSILON_INTERPOLATION in sprite.py
        return None, None, None, None
    
    # We're going to use cross_a values to find the most eggregious crossing, so
    # set the non-crossing entries to -Inf. This is less cumbersome to implement
    # than setting to NaN and use np.nanargmin, because some slices in np.argmax
    # below might be all -Inf.
    cross_a[np.logical_not(crossings)] = -np.inf
    abs_cross_a = np.abs(1. - cross_a)

    # Get the crossing closest to the current vertex position for each vertex in
    # sprite_0. This will either be the most recent crossing or the most
    # imminent crossing.
    inds_crossings = np.argmin(abs_cross_a, axis=1)
    crossing_points = np.array([
        traj[i_0, 0] + cross_a[i_0, i_1] * (traj[i_0, 1] - traj[i_0, 0])
        for (i_0, i_1) in enumerate(inds_crossings)
    ])

    # Get segment from crossing point to trajectory end
    diffs_from_crossings = np.array([
        traj[i, 1] - crossing_points[i]
        for i in range(len(crossing_points))
    ])

    # Now we try to deduce which of the crossing points is the true collision
    # point. We do this by selecting the one for which the trajectory segment
    # endpoint is furthest from the crossing point, i.e. the point that is
    # deepest inside sprite_1.
    dists_vertices_crossings = np.linalg.norm(diffs_from_crossings, axis=1)
    dists_vertices_crossings[dists_vertices_crossings == np.inf] = 0
    crossing_points_ind = np.argmax(dists_vertices_crossings)
    sprite_1_ind = inds_crossings[crossing_points_ind]
    collision_point = crossing_points[crossing_points_ind]
    since_collision = diffs_from_crossings[crossing_points_ind]

    # If the true collision point is in the future, don't collide.
    if cross_a[crossing_points_ind, sprite_1_ind] > 1:
        # True collision point is in the future
        return collision_point, np.nan, since_collision, None

    # Finally, compute the collision vector normal to sprite_1
    delta_vertex_1 = (vertices_1_next[sprite_1_ind] -
                      vertices_1[sprite_1_ind])
    normal_vector = np.array([delta_vertex_1[1], -1 * delta_vertex_1[0]])
    collision_normal = normal_vector / np.linalg.norm(normal_vector)

    # Compute the perpendicular margin of overlap
    projection = delta_vertex_1 * (
        np.dot(since_collision, delta_vertex_1) /
        np.dot(delta_vertex_1, delta_vertex_1)
    )
    perpendicular = since_collision - projection

    return collision_point, collision_normal, since_collision, perpendicular


def _get_collision_vectors(sprite_0, sprite_1, delta_t):
    """Get contact point, collision normal, and displacement since collision.
    
    This function produces the collision point, collision normal, and relative
    displacement since the collision event for two sprites. It approximately
    infers these as  if the sprites had continuous time, i.e. the collision
    point is the true point of collision between the previous and current
    timestep.

    The collision point, normal, and displacement (plus things like the sprite
    centers of mass and moments of inertia) of are all that are needed from the
    sprite geometry for computing the physics. Once these are computed, we
    never have to look at the sprite vertices again.

    Args:
        sprite_0: Instance of ../sprite.Sprite.
        sprite_1: Instance of ../sprite.Sprite.
        delta_t: Scalar. Timestep of the physics.

    Returns:
        collision_point: None or numpy array of shape [2]. Position of the
            collision.
        collision_normal: None or numpy array of shape [2]. Zero-centered vector
            indicating the normal of the collision surface, i.e. the direction
            in which the collision force is applied. It is signed, so we let it
            point toward the sprite_0 side of the collision.
        since_collision: None or numpy array of shape [2]. Relative displacement
            of the two sprites since the collision. It is signed, so we let it
            point toward the sprite_1 side of the collision, i.e. representing
            how much sprite_0 has moved relative to sprite_1 since the collision
            event.
    """
    # First find the collision point on sprite_0 boundary
    collision_point_0, collision_normal_0, since_collision_0, perp_0 = (
        _directed_collision_vectors(sprite_1, sprite_0, delta_t))
    
    # Now find the collision point on sprite_1 boundary
    collision_point_1, collision_normal_1, since_collision_1, perp_1 = (
        _directed_collision_vectors(sprite_0, sprite_1, delta_t))

    # Make since_collision_i zero if no collision happened in the i direction
    if collision_point_0 is None:
        since_collision_0 = np.zeros(2)
    else:
        # Must negate collision_normal and since_collision for symmetry
        collision_normal_0 = -1. * collision_normal_0
        since_collision_0 = -1. * since_collision_0
    if since_collision_1 is None:
        since_collision_1 = np.zeros(2)

    # Pick which collision point to treat as the real collision point
    if np.linalg.norm(since_collision_0) > np.linalg.norm(since_collision_1):
        return collision_point_0, collision_normal_0, since_collision_0, perp_0
    else:
        return collision_point_1, collision_normal_1, since_collision_1, perp_1


def _collide_without_update_angle_vel(sprite_0,
                                      sprite_1,
                                      collision_point,
                                      collision_normal,
                                      elasticity=1.,
                                      symmetric=True):
    """Simulate collision in which sprite angular velocity cannot change.

    Since the sprite angular velocity is prohibited from changing, this
    collision is non-Newtonion, i.e. does not conserve angular momentum. However
    it does converse kinetic enery and momentum, hence models what the collision
    would be if the collision point were between the two sprites' centers of
    mass.

    Args:
        sprite_0: Instance of ../sprite.Sprite. If symmetric = False, this
            sprite is the one that is doing the colliding, i.e. this one's
            velocity changes.
        sprite_1: Instance of ../sprite.Sprite. If symmetric = False, this
            sprite is treated as having infinite mass (i.e. has fixed velocity).
        collision_point: Numpy array of shape [2]. Position of the collision.
        collision_normal: Numpy array of shape [2]. Unit normal of the
            collision, pointing towards sprote_0. Must have norm 1.
        elasticity: Float in [0, 1]. Elasticity 1 means the sprites bounce
            fully. Elasticity 0 means they stick together completely.
        symmetric: Bool. If True, collision is symmetric and both sprites'
            velocities are updated. If False, sprite_1 is treated as having
            infinite mass.
    """
    # Sanity check that collision_normal has norm 1
    collision_normal_norm = np.linalg.norm(collision_normal)
    if not np.isclose(collision_normal_norm, 1., atol=1e-4):
        raise ValueError(
            'collision_normal_norm is {}, which is not close to 1.'.format(
                collision_normal_norm))

    vel_0 = sprite_0.velocity
    vel_1 = sprite_1.velocity
    m_0 = sprite_0.mass
    m_1 = sprite_1.mass

    # Only consider the component of velocity parallel to normal
    vel_0_normal = np.dot(vel_0, collision_normal) * collision_normal
    vel_1_normal = np.dot(vel_1, collision_normal) * collision_normal

    # Find inertial reference frame, i.e. velocity of center of mass of the
    # entire system
    if symmetric:
        vel_cm_normal = (vel_0_normal * m_0 + vel_1_normal * m_1) / (m_0 + m_1)
    else:
        vel_cm_normal = vel_1_normal

    # Reflect each velocity across vel_cm, with elasticity
    delta_v_0 = (1 + elasticity) * (vel_cm_normal - vel_0_normal)
    delta_v_1 = (1 + elasticity) * (vel_cm_normal - vel_1_normal)

    # Update the sprite velocities
    sprite_0.velocity += delta_v_0
    sprite_1.velocity += delta_v_1


def _collide_with_update_angle_vel(sprite_0,
                                   sprite_1,
                                   collision_point,
                                   collision_normal,
                                   elasticity=1.,
                                   symmetric=True):
    """Simulate collision.

    This simulates Newtonian collisions between sprite_0 and sprite_1. It takes
    into account the geometry of the sprites, collision contact point, collision
    normal at the contact point, rotational inertia of the sprites, etc. Hence
    the collisions this simulates are very realistic.

    Deriving the equations governing the collision dynamics is non-trivial, so
    we sketch the approach here:
        
        First, for any collision between a rigid body with moment of unertia I,
        angular velocity w, mass M, and center-of-mass velocity v, we have
            I delta(w) = M delta(v) r sin(theta)
        where r is the distance between the collision point and the center of
        mass and theta is the angle between the collision normal and the center
        of mass. This equation is independent of the object with which the
        collision happens, and can be derived from conservation of momentum and
        conservation of angular momentum given some point mass being collided
        with.

        Now suppose we have two rigid bodies with subscripts 0 and 1. Then the
        above equation gives us
            I_0 delta(w_0) = M_0 delta(v_0) r_0 sin(theta_0)
            I_1 delta(w_1) = M_1 delta(v_1) r_1 sin(theta_1)
        Now conversation of global momentum gives us
            0 = sum_i[M_i delta(v_i)]
        And conservation of global kinetic energy gives
            0 = sum_i[M_i (v_i_after^2 - v_i_before^2) +
                      I_i (w_i_after^2 - w_i_before^2)]
              = sum_i[M_i delta(v_i) gamma(v_i) + I_i delta(w_i) gamma(w_i)]
        where gamma(x_i) = x_i_after + x_i_before.
        
        We must now solve these four equations for the four variable
            {v_i_after, w_i_after}_{i = 0, 1}
        Substituting gamma(x_i) = delta(x_i) + 2 x_i_before into the kinetic
        energy equation and eliminating all appearances of delta(v_1) and
        delta(w_1) using the previous equations, we end up with the solution in
        the code.

    Args:
        sprite_0: Instance of ../sprite.Sprite. If symmetric = False, this
            sprite is the one that is doing the colliding, i.e. this one's
            velocity changes.
        sprite_1: Instance of ../sprite.Sprite. If symmetric = False, this
            sprite is treated as having infinite mass and infinite moment of
            inertia (i.e. has fixed velocity and angular velocity).
        collision_point: Numpy array of shape [2]. Position of the collision.
        collision_normal: Numpy array of shape [2]. Unit normal of the
            collision, pointing towards sprote_0. Must have norm 1.
        elasticity: Float in [0, 1]. Elasticity 1 means the sprites bounce
            fully. Elasticity 0 means they stick together completely.
        symmetric: Bool. If True, collision is symmetric and both sprites'
            velocities are updated. If False, sprite_1 is treated as having
            infinite mass and infinite moment of inertia.
    """

    # Alias some scalars that we'll need
    m_0 = sprite_0.mass
    m_1 = sprite_1.mass
    w_0 = sprite_0.angle_vel
    w_1 = sprite_1.angle_vel
    i_0 = sprite_0.moment_of_inertia
    i_1 = sprite_1.moment_of_inertia

    # Extract only magnitude of velocity component parallel to normal
    v_0 = np.dot(sprite_0.velocity, collision_normal)
    v_1 = np.dot(sprite_1.velocity, collision_normal)

    # Extract sin(theta), where theta is angle between normal and contact point
    c_point_0 = collision_point - sprite_0.position
    c_point_1 = collision_point - sprite_1.position
    r_0 = np.linalg.norm(c_point_0)
    r_1 = np.linalg.norm(c_point_1)
    sin_theta_0 = np.cross(c_point_0, collision_normal) / r_0
    sin_theta_1 = np.cross(c_point_1, collision_normal) / r_1

    # Apply the collision equations. See docstring for a derivation sketch.
    s_0 = r_0 * sin_theta_0
    s_1 = r_1 * sin_theta_1

    a = m_0 + m_1 + m_0 * m_1 * ((s_0 * s_0 / i_0) + (s_1 * s_1 / i_1))
    b = (1 + elasticity) * (v_0 - v_1 + w_0 * s_0 - w_1 * s_1)
    if symmetric:
        delta_v_0 = -1 * m_1 * b / a
        delta_v_1 = m_0 * b / a
    else:
        delta_v_0 = -1 * m_1 * b / (a - m_0)
        delta_v_1 = 0.
    delta_w_0 = m_0 * delta_v_0 * s_0 / i_0
    delta_w_1 = m_1 * delta_v_1 * s_1 / i_1

    # Update sprite velocities and angular velocities
    sprite_0.velocity += delta_v_0 * collision_normal
    sprite_1.velocity += delta_v_1 * collision_normal
    sprite_0.angle_vel += delta_w_0
    sprite_1.angle_vel += delta_w_1


class Collision(abstract_force.AbstractForce):
    """Collision simulator.
    
    This class simulates Newtonian collisions between two rigid bodies, namely
    sprites in the environment. The physics assumes that the sprites have
    uniform density, as this is important for calculating their moments of
    inertia (see ../sprite for details).
    """

    def __init__(self, elasticity=1., symmetric=False, update_angle_vel=True,
                 max_recursion_depth=0):
        """Constructor.
        
        Args:
            elasticity: Float in [0, 1]. Elasticity 1 means the sprites bounce
                fully. Elasticity 0 means they stick together completely.
            symmetric: Bool. If True, collision is symmetric and both sprites
                are updated. If False, sprite_1 in .step() is treated as having
                infinite mass. This is useful for modeling bounces off of
                obstructors/walls that are fixed in the environment.
            update_angle_vel: Bool. If True, fully simulate the rotational
                mechanics and update the sprites angular velocity. If False,
                sprite angular velocities are not updated and the collision is
                treated as if the contact point was between the two sprites'
                centers of mass. Simulating the full rotational mechanics is
                slightly slower.
            max_recursion_depth: Int. Number of times self.step() may recurse.
                Usually, 0 (no recursion) is fine. However, larger values make
                the collisions slightly more accurate (at the expense of runtime
                efficiency) if multiple collisions are happening within a single
                step, e.g. colliding into a corner.
        """
        self._elasticity = elasticity
        self._symmetric = symmetric
        self._update_angle_vel = update_angle_vel
        self._max_recursion_depth = max_recursion_depth

    def step(self, sprite_0, sprite_1, updates_per_env_step, recursion_depth=0):
        """Step the physics.
        
        Args:
            sprite_0: Instance of ../sprite.Sprite. If self._symmetric = False,
                this sprite is the one that is doing the colliding, i.e. this
                one's velocity changes.
            sprite_1: Instance of ../sprite.Sprite. If self._symmetric = False,
                this sprite is treated as having infinite mass.
            updates_per_env_step: Int. Number of times this force step is called
                for each step of the physics in the environment. This is used
                here for inferring the exact contact point in the collision.
            recursion_depth: Int. Number of times this function has recursed.
                Used internally only to catch and avoid max recursion depth
                errors.
        """
        if recursion_depth > self._max_recursion_depth:
            return

        if sprite_0 == sprite_1:
            return

        if not sprite_0.overlaps_sprite(sprite_1):
            return

        delta_t = 1. / updates_per_env_step

        # First get collision point, normal, and relative sprite displacement
        # since the collision
        collision_point, collision_normal, _, perpendicular = (
            _get_collision_vectors(sprite_0, sprite_1, delta_t))
        
        if collision_point is None:
            # Although the sprites do overlap, neither sprite contains any of
            # the other's vertices. This can happen when the sprites collide
            # exactly at two corners. This is annoying to handle because we must
            # infer which corner hit which face in between timesteps, but must
            # be done to ensure stability. See self._make_disjoint() for
            # details.
            self._make_disjoint(sprite_0, sprite_1)
        else:
            # Whether the collision point is in the future, in which case we
            # leave the sprites alone.
            future_collision_point = (
                np.isscalar(collision_normal) and np.isnan(collision_normal))

            if not future_collision_point:
                # There was a collision in the past, so we must simulate it
                # First, displace the sprites so they are no longer intersecting
                # Note that instead of perpendicular displacement we could use
                # since_collision. However, sometimes since_collision can have a
                # very acute angle and large magnitude due to angular velocity
                # or multiple collisions, so in prectice displacing by the
                # perpendicular is more stable.
                if self._symmetric:
                    sprite_0.position = (
                        sprite_0.position - (0.5 + _EPSILON) * perpendicular)
                    sprite_1.position = (
                        sprite_1.position + (0.5 + _EPSILON) * perpendicular)
                else:
                    sprite_0.position = (
                        sprite_0.position - (1. + _EPSILON) * perpendicular)

                # Second, change sprite velocities and angular velocities per
                # Newtonian physics
                if self._update_angle_vel:
                    _collide_with_update_angle_vel(
                        sprite_0,
                        sprite_1,
                        collision_point,
                        collision_normal,
                        elasticity=self._elasticity,
                        symmetric=self._symmetric,
                    )
                else:
                    _collide_without_update_angle_vel(
                        sprite_0,
                        sprite_1,
                        collision_point,
                        collision_normal,
                        elasticity=self._elasticity,
                        symmetric=self._symmetric,
                    )
            else:
                return
            
        # After perturbing the sprite positions and velocities, we recursively
        # step again in case that perturbation has now created another
        # collision.
        self.step(sprite_0, sprite_1, updates_per_env_step,
                  recursion_depth=recursion_depth + 1)

    def _make_disjoint(self, sprite_0, sprite_1):
        """Perturb the positions of sprite_0 and sprite_1 to make them disjoint.

        In theory, with infinitesimal delta_t, the collision detection would
        work without this correction since every collision occurs by a vertex of
        one sprite entering another sprite. However, with a non-zero delta_t, if
        a collision occurs almost exactly at two sharp corners of sprites, the
        vertex entrypoint could be missed due to the time discretization. So
        this correction will catch that case and perturb the sprites so they no
        longer intersect.

        Furthermore, sometimes if collisions with multiple sprites happen within
        one physics timestep, the collision misses one of them, so this
        perturbation comes in handy then to essentially push one of the
        collisions off until the next timestep.

        Finally, sometimes a sprite position may be macigally changed (e.g. by a
        discrete action space) to make it enter another sprite without moving
        there by its physics, so this correction is useful to separate the two
        overlapping sprites in that case as well.

        The way this function works is a bit ugly and not easy to describe. If
        you want to understand it, please draw out some examples on paper and
        see what the code does with them.

        Args:
            sprite_0: Instance of ../sprite.Sprite.
            sprite_1: Instance of ../sprite.Sprite.

        Returns:
            Boolean indicating whether the sprites were overlapping. This is
            useful for the calling code to know whether the sprie positions have
            been perturbed.
        """
        crossing_points, inds_crossings = sprite_lib.sprite_edge_crossings(
            sprite_0, sprite_1)

        if len(inds_crossings) <= 1:
            return True

        # correction_i is how much sprite_i must move (independent of sprite_j)
        # for the two to be disjoint.
        correction_0 = _position_correction(
            crossing_points,
            sprite_0,
            inds_crossings[:, 0],
            sprite_1,
            inds_crossings[:, 1])
        correction_1 = _position_correction(
            crossing_points,
            sprite_1,
            inds_crossings[:, 1],
            sprite_0,
            inds_crossings[:, 0])

        if np.linalg.norm(correction_0) > np.linalg.norm(correction_1):
            correction = -1 * (1 + _EPSILON) * correction_0
        else:
            correction = (1 + _EPSILON) * correction_0
        
        if not all(np.isfinite(correction)):  # no correction needed
            correction = np.zeros(2)
        
        if self._symmetric:
            sprite_0.position = sprite_0.position + 0.5 * correction
            sprite_1.position = sprite_1.position - 0.5 * correction
        else:
            sprite_0.position = sprite_0.position + correction
        
        return False


def _position_correction(crossing_points,
                         sprite_0,
                         crossing_inds_0,
                         sprite_1,
                         crossing_inds_1):
    """Finds a correction for the position to make the sprites disjoint.
    
    This finds how much the sprite_0 must move to alleviate the crossing points
    closest to its center of mass.

    Args:
        crossing_points: Numpy float array of size [K, 2].
        sprite_0: Instance of ../sprite.Sprite.
        crossing_inds_0: Numpy int array of size [K, 2].
        sprite_1: Instance of ../sprite.Sprite.
        crossing_inds_1: Numpy int array of size [K, 2].
    """
    vertices = sprite_0.vertices
    other_vertices = sprite_1.vertices

    # Sort crossing points by their proximity to the sprite_0's center of mass.
    dists_from_cm = np.linalg.norm(crossing_points - sprite_0.position, axis=1)
    sorted_inds = np.argsort(dists_from_cm)

    if sorted_inds[0] == sorted_inds[1]:
        # sprite_0 has no offending vertices, so sees no correction itself
        return np.array([np.inf, np.inf])

    # Get the two closest points. Ultimately we will find a perturbation
    # perpendicular to the segment between these two points. Specifically, we
    # will find the sprite_0 vertex closest to the other sprite_0's center of
    # mass along the direction perpendicular to the segment between these two
    # points. That vertex is the most eggregious offender and will tell use how
    # much we have to perturb.
    pt_0 = crossing_points[sorted_inds[0]]
    pt_1 = crossing_points[sorted_inds[1]]

    def _get_norm_v(point_0, point_1):
        # Get normalized vector perpendicular to [point_0, point_1] and pointing
        # towards sprite_1.position
        normalized_bdry = point_1 - point_0
        normalized_bdry /= np.linalg.norm(normalized_bdry)
        norm_v = normalized_bdry * -1 * np.sign(
            np.dot(sprite_1.position - point_0, normalized_bdry))
        return norm_v

    # Get the normal vector from the [pt_0, pt_1] segment in the direction of
    # cm_other.
    if crossing_inds_1[sorted_inds[0]] == crossing_inds_1[sorted_inds[0]]:
        # In this case using the edge of sprite_1 is more accurate
        pt_0_ind = crossing_inds_1[sorted_inds[0]]
        pt_1_ind = (pt_0_ind - 1) % len(other_vertices)
        norm_v = _get_norm_v(other_vertices[pt_1_ind], other_vertices[pt_0_ind])
    else:
        norm_v = _get_norm_v(pt_0, pt_1)

    # We're now going to traverse through some of sprite_0's vertices looking
    # for the one furthest in the norm_v direction. However, we don't want to
    # consider all of sprite_0's vertices (that could give wrong results for
    # some star-shaped sprites). We only want to consider those that lie between
    # the crossing points of interest. So we do this by finding which endpoint
    # of the first crossing point's sprite_0 edge lies in the norm_v direction
    # and traverse from there until we are in the negative norm_v direction.
    ind_forward = crossing_inds_0[sorted_inds[0]]
    ind_backward = (ind_forward - 1) % len(vertices)
    if np.dot(vertices[ind_forward] - pt_0, norm_v) > 0:
        # The forward endpoint of this sprite_0 edge is in the norm_v direction
        parity = 1
        current_ind = ind_forward
    elif np.dot(vertices[ind_backward] - pt_0, norm_v) > 0:
        # The backward endpoint of this sprite_0 edge is in the norm_v direction
        parity = -1
        current_ind = ind_backward
    else:
        # Neither vertex is in the norm_v direction, in which case we aren't
        # actually overlapping the other sprite_0 (probably just barely touching
        # it, perhaps caused by _EPSILON)
        return np.array([np.inf, np.inf])

    # Now iterate through vertices looking for the most eggregious offender
    # until we're no longer in the norm_v direction.
    worst_penalty = 0
    while np.dot(vertices[current_ind] - pt_0, norm_v) > 0:
        penalty = np.dot(vertices[current_ind] - pt_0, norm_v)
        if penalty > worst_penalty:
            worst_penalty = penalty
        current_ind = (current_ind + parity) % len(vertices)
    
    correction = worst_penalty * norm_v

    return correction
