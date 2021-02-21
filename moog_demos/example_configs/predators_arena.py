"""Task with predators chasing agent in open arena.

The predators (red circles) chase the agent. The predators bouce off the arena
boundaries, while the agent cannot exit but does not bounce (i.e. it has
inelastic collisions with the boundaries). Trials only terminate when the agent
is caught by a predator. The subject controls the agent with a joystick.

This task also contains an auto-curriculum: When the subject does well (evades
the predators for a long time before being caught), the predators' masses are
decreased, thereby increasing the predators' speeds. Conversely, when the
subject does poorly (gets caught quickly), the predators' masses are increased,
thereby decreasing the predators' speeds.
"""

import collections
import numpy as np
import os

from moog import action_spaces
from moog import game_rules
from moog import observers
from moog import physics as physics_lib
from moog import shapes
from moog import tasks
from moog.state_initialization import distributions as distribs
from moog.state_initialization import sprite_generators


class StateInitialization():
    """State initialization class to dynamically adapt predator mass.
    
    This is essentially an auto-curriculum: When the subject does well (evades
    the predators for a long time before being caught), the predators' masses
    are decreased, thereby increasing the predators' speeds. Conversely, when
    the subject does poorly (gets caught quickly), the predators' masses are
    increased, thereby decreasing the predators' speeds.
    """

    def __init__(self, num_predators, step_scaling_factor, threshold_trial_len):
        """Constructor.

        This class uses the meta-state to keep track of the number of steps
        before the agent is caught. See the game rules section near the bottom
        of this file for the counter incrementer.

        Args:
            step_scaling_factor: Float. Fractional decrease of predator mass
                after a trial longer than threshold_trial_len. Also used as
                fractional increase of predator mass after a trial shorter than
                threshold_trial_len. Should be small and positive.
            threshold_trial_len: Length of a trial above which the predator
                mass is decreased and below which the predator mass is
                increased.
        """
        self._mass = 1.
        self._step_scaling_factor = step_scaling_factor
        self._threshold_trial_len = threshold_trial_len

        # Agent
        agent_factors = distribs.Product(
            [distribs.Continuous('x', 0., 1.),
            distribs.Continuous('y', 0., 1.)],
            shape='circle', scale=0.1, c0=0.33, c1=1., c2=0.66,
        )
        self._agent_generator = sprite_generators.generate_sprites(
            agent_factors, num_sprites=1)

        # Predators
        predator_factors = distribs.Product(
            [distribs.Continuous('x', 0., 1.),
            distribs.Continuous('y', 0., 1.)],
            shape='circle', scale=0.1, c0=0., c1=1., c2=0.8,
        )
        self._predator_generator = sprite_generators.generate_sprites(
            predator_factors, num_sprites=num_predators)

        # Walls
        self._walls = shapes.border_walls(
            visible_thickness=0., c0=0., c1=0., c2=0.5)

        self._meta_state = None

    def state_initializer(self):
        """State initializer method to be fed to environment."""
        agent = self._agent_generator(without_overlapping=self._walls)
        predators = self._predator_generator(
            without_overlapping=self._walls + agent)

        if self._meta_state is not None:
            if self._meta_state['step_count'] > self._threshold_trial_len:
                self._mass -= self._mass * self._step_scaling_factor
            else:
                self._mass += self._mass * self._step_scaling_factor
        for s in predators:
            s.mass = self._mass
        
        state = collections.OrderedDict([
            ('walls', self._walls),
            ('agent', agent),
            ('predators', predators),
        ])
        return state

    def meta_state_initializer(self):
        """Meta-state initializer method to be fed to environment."""
        self._meta_state = {'step_count': 0}
        return self._meta_state


def get_config(num_predators):
    """Get config dictionary of kwargs for environment constructor.
    
    Args:
        num_predators: Int. Number of predators.
    """

    ############################################################################
    # Sprite initialization
    ############################################################################

    state_initialization = StateInitialization(
        num_predators=num_predators,
        step_scaling_factor=0.1,
        threshold_trial_len=200,
    )

    ############################################################################
    # Physics
    ############################################################################

    agent_friction_force = physics_lib.Drag(coeff_friction=0.25)
    predator_friction_force = physics_lib.Drag(coeff_friction=0.04)
    predator_random_force = physics_lib.RandomForce(max_force_magnitude=0.03)
    predator_attraction = physics_lib.DistanceForce(
        physics_lib.linear_force_fn(zero_intercept=-0.0025, slope=0.0001))
    elastic_asymmetric_collision = physics_lib.Collision(
        elasticity=1., symmetric=False)
    inelastic_asymmetric_collision = physics_lib.Collision(
        elasticity=0., symmetric=False)
    
    forces = (
        (agent_friction_force, 'agent'),
        (predator_friction_force, 'predators'),
        (predator_random_force, 'predators'),
        (predator_attraction, 'agent', 'predators'),
        (elastic_asymmetric_collision, 'predators', 'walls'),
        (inelastic_asymmetric_collision, 'agent', 'walls'),
    )
    
    physics = physics_lib.Physics(*forces, updates_per_env_step=10)

    ############################################################################
    # Task
    ############################################################################

    task = tasks.ContactReward(
        -1, layers_0='agent', layers_1='predators', reset_steps_after_contact=0)

    ############################################################################
    # Action space
    ############################################################################

    action_space = action_spaces.Joystick(
        scaling_factor=0.01, action_layers='agent')

    ############################################################################
    # Observer
    ############################################################################

    observer = observers.PILRenderer(
        image_size=(64, 64), anti_aliasing=1, color_to_rgb='hsv_to_rgb')

    ############################################################################
    # Game rules
    ############################################################################

    def _increment_count(meta_state):
        meta_state['count'] += 1
    rules = game_rules.ModifyMetaState(_increment_count)

    ############################################################################
    # Final config
    ############################################################################

    config = {
        'state_initializer': state_initialization.state_initializer,
        'physics': physics,
        'task': task,
        'action_space': action_space,
        'observers': {'image': observer},
        'meta_state_initializer': state_initialization.meta_state_initializer,
    }
    return config
