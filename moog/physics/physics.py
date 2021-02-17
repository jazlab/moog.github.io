"""Physics class.

The `physics` argument to ..environment.Environment is typically an instance of
the Physics class in this file.
"""

from . import abstract_physics
import itertools
import numpy as np


class Physics(abstract_physics.AbstractPhysics):
    """Force physics class."""

    def __init__(self, *forces, updates_per_env_step=1, corrective_physics=()):
        """Constructor.

        Examples of usage:

            ```python
            friction_force = Friction(coeff_friction=1)
            spring_force = Spring(equilibrium=0.3, spring_const=0.1)
            collision_force = Collision(elasticity=0.5)

            # Specify only friction on the avatar sprite(s)
            force = (friction_force, 'avatar')
            my_physics = Physics(force)

            # Specify friction on the avatar and prey sprite(s)
            force = (friction_force, ['avatar', 'prey'])
            my_physics = Physics(force)

            # Specify friction on avatar and springs between avatar and all prey
            forces = (
                (friction_force, 'avatar'),
                (spring_force, 'avatar', 'prey'),
            )
            my_physics = Physics(*forces)

            # Add collisions between all avatar/prey sprites and walls/neutrals
            forces = (
                (friction_force, 'avatar'),
                (spring_force, 'avatar', 'prey'),
                (collision_force, ['avatar', 'prey'], ['walls', 'neutrals']),
            )
            my_physics = Physics(*forces)
            ```

        Args:
            *forces: Iterable. Each element is a tuple (force_instance, *args)
                where force_instance is an instance of
                abstract_force.AbstractForce and *args has the same length as
                the number of arguments of force_instance's step() method. Each
                element of args may be either a string key of the environment
                state or an iterable of such. This specifies all sprites to feed
                into the force.
            updates_per_env_step: Int. Number of physics applications per step
                of the environment. A higher value means smoother physics but
                slower runtime. This is fed into the individual forces so that
                they can accomodate their force strength accordingly.
            corrective_physics: Optional instance (or iterable of instances) of
                abstract_physics.AbstractPhysics to be applied every step before
                updating the sprite positions. This is typically used to apply
                corrections to sprite velocities. For example, it can be used to
                enforce rigid tethers between sprites. 
        """
        super(Physics, self).__init__(updates_per_env_step=updates_per_env_step)
        self._forces = forces
        if not isinstance(corrective_physics, (list, tuple)):
            corrective_physics = [corrective_physics]
        self._corrective_physics = corrective_physics
    
    def reset(self, state):
        for force in self._forces:
            force[0].reset(state)
        
        for corrective_physics in self._corrective_physics:
            corrective_physics.reset(state)

    def _args_to_iterables(self, args_iterables):
        """Converts all non-iterable elements to singleton tuples."""
        args_iterables = [
            x if isinstance(x, (tuple, list)) else (x,)
            for x in args_iterables
        ]
        return args_iterables

    def apply_physics(self, state, updates_per_env_step):
        """Move the sprites according to the physics."""

        # Update sprites based on forces
        for (force, *args_iterables) in self._forces:
            args_iterables = self._args_to_iterables(args_iterables)

            # If args_iterables = [['avatar', 'prey'], ['walls', 'neutrals']],
            # then
            # args_combinations = [
            #   ['avatar', 'walls'],
            #   ['avatar', 'neutrals'],
            #   ['prey', 'walls'],
            #   ['prey', 'neutrals'],
            # ]
            args_combinations = itertools.product(*args_iterables)

            for args in args_combinations:
                for sprites in itertools.product(*[state[s] for s in args]):
                    force.step(
                        *sprites, updates_per_env_step=updates_per_env_step)

        for corrective_physics in self._corrective_physics:
            corrective_physics.apply_physics(state, updates_per_env_step)

        # Move sprites based on their velocity
        delta_t = 1. / updates_per_env_step
        for layer in state:
            for sprite in state[layer]:
                sprite.update_pos_from_vel(delta_t=delta_t)
