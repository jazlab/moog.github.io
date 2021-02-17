"""Rigid tether physics.

The two classes Tether and TetherZippedLayers can be used to rigidly tether
together sets of sprites, optionally also tethering them to an anchor point in
the environment.

These classes are typically used as corrective physics for a physics.Physics
instance.
"""

from . import abstract_physics
import itertools
import numpy as np


def _change_rotation_coordinates(sprite,
                                 origin,
                                 origin_velocity,
                                 updates_per_env_step):
    """Change rotation coordinates to new origin.
    
    This function gets the angular momentum, moment of inertia, radius, and
    perpendicular relative to a new origin point.
    """
    # Get angular momentum coming from velocity
    delta_position = sprite.velocity / updates_per_env_step
    parallel = (sprite.position + 0.5 * delta_position) - origin
    radius = np.linalg.norm(parallel)
    parallel /= radius
    perpendicular = np.matmul(np.array([[0, -1], [1, 0]]), parallel)
    perp_vel = np.dot(sprite.velocity - origin_velocity, perpendicular)
    angular_momentum = perp_vel * sprite.mass * radius

    # Add angular momentum coming from angular velocity
    angular_momentum += sprite.angle_vel * sprite.moment_of_inertia

    # Get moment of inertia
    moment_of_inertia = sprite.moment_of_inertia + sprite.mass * radius * radius

    return angular_momentum, moment_of_inertia, radius, perpendicular


def _tether_sprites(sprites,
                    updates_per_env_step,
                    update_angle_vel=True,
                    anchor=None):
    """Apply a tether to a set of sprites."""
    # Can't tether if no sprites
    if len(sprites) == 0:
        return
        
    # Compute center of mass
    total_mass = sum([s.mass for s in sprites])
    
    # Can't tether if infinite masses are involved
    if np.isinf(total_mass):
        return
    
    center_of_mass = (
        sum([s.mass * s.position for s in sprites]) / total_mass)

    # Compute total momentum and velocity
    total_momentum = sum([s.mass * s.velocity for s in sprites])
    total_velocity = total_momentum / total_mass

    if anchor is not None:
        center_of_mass = anchor
        total_velocity = np.zeros(2)

    if update_angle_vel:
        # Compute total angular momentum and angular velocity
        angular_momenta, moments_of_inertia, radii, perpendiculars = zip(*[
            _change_rotation_coordinates(
                s, center_of_mass, total_velocity, updates_per_env_step)
            for s in sprites
        ])
        total_angular_momentum = sum(angular_momenta)
        total_moment_of_inertia = sum(moments_of_inertia)
        total_angular_velocity = (
            total_angular_momentum / total_moment_of_inertia)

        # Set velocities and angular velocities
        for s, radius, perp in zip(sprites, radii, perpendiculars):
            s.velocity = (
                total_velocity + radius * perp * total_angular_velocity)
            s.angle_vel = total_angular_velocity
    else:
        # Set velocities and angular velocities
        for s in sprites:
            s.velocity = total_velocity
            s.angle_vel = 0.


class Tether(abstract_physics.AbstractPhysics):
    """Rigid tether physics class.
    
    This is used to rigidly tether all sprites in specified layers. For example,
    if you want all prey to be rigidly connected, use Tether('prey'). It is
    typically used as a corrective_physics argument to physics.Physics.

    WARNING: This rigid tether does not work if sprites have their positions
    changed directly --- it only works to adjust sprite velocities and angular
    velocities. This can sometimes have noticeable effects if tethered sprites
    have some collisions.
    """

    def __init__(self, layer_names, update_angle_vel=True, anchor=None):
        """Constructor.
        
        Args:
            layer_names: String or iterable of strings. All elements must be
                keys in environment state. All sprites in all of these layers
                will be tethered and will move together.
            update_angle_vel: Bool. If True, fully simulate the rotational
                mechanics and update the sprites angular velocity.
            anchor: Optional anchor point. If given, sprites will be tethered to
                this fixed point and will remain at a fixed distance from it.
        """
        if not isinstance(layer_names, (list, tuple)):
            self._layer_names = [layer_names]
        else:
            self._layer_names = layer_names
        self._update_angle_vel = update_angle_vel
        self._anchor = anchor

    def apply_physics(self, state, updates_per_env_step):
        """Step the physics.
        
        Args:
            sprites: Environment state. Dictionary of iterables of instances of
                ../sprite.Sprite.
            updates_per_env_step: Int. Number of times this physics is called
                per environment step. Unused in this method, but needed to
                satisfy the AbstractPhysics signature.
        """
        sprites = list(itertools.chain(
            *[state[layer_name] for layer_name in self._layer_names]))
        _tether_sprites(
            sprites, updates_per_env_step,
            update_angle_vel=self._update_angle_vel, anchor=self._anchor)


class TetherZippedLayers(abstract_physics.AbstractPhysics):
    """Apply rigid tethers between zipped sprites across layers.
    
    This zips the sprites in different layers and tethers them together.
    Specifically, for each index i, the set {i'th sprite in each layer} is
    tethered together, assuming all provided layers have the same number of
    sprites.
    
    The is useful for example if you want each sprite in a layer to carry a
    local occluder sprite with it, as in spatial delayed match-to-sample.

    WARNING: As with Tether above, this class does not work if sprites have
    their positions changed directly --- it only works to adjust sprite
    velocities and angular velocities.
    """

    def __init__(self, layer_names, update_angle_vel=True, anchor=None):
        """Constructor.
        
        Args:
            layer_names: String or iterable of strings. All elements must be
                keys in environment state. Sprites these layers will be zipped
                and tethered. These layers must all have the same number of
                sprites.
            update_angle_vel: Bool. If True, fully simulate the rotational
                mechanics and update the sprites angular velocity.
            anchor: Optional anchor point. If given, sprites will be tethered to
                this fixed point and will remain at a fixed distance from it.
        """
        if not isinstance(layer_names, (list, tuple)):
            self._layer_names = [layer_names]
        else:
            self._layer_names = layer_names
        self._update_angle_vel = update_angle_vel
        self._anchor = anchor

    def apply_physics(self, state, updates_per_env_step):
        """Step the physics.
        
        Args:
            sprites: Environment state. Dictionary of iterables of instances of
                ../sprite.Sprite.
            updates_per_env_step: Int. Number of times this physics is called
                per environment step. Unused in this method, but needed to
                satisfy the AbstractPhysics signature.
        """
        # Ensure that all layers have the same number of sprites
        layer_sprites = [state[layer_name] for layer_name in self._layer_names]
        layer_lengths = [len(x) for x in layer_sprites]
        if not all(length == layer_lengths[0] for length in layer_lengths):
            raise ValueError(
                'All layers fed into TetherAcrossLayers must have the same '
                'number of sprites, but their counts are {}'.format(
                    layer_lengths))
        
        for sprites in zip(*layer_sprites):
            _tether_sprites(
                sprites, updates_per_env_step,
                update_angle_vel=self._update_angle_vel, anchor=self._anchor)
