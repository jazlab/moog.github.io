"""Task with predators chasing agent in open arena.

The predators (red circles) chase the agent. The predators bouce off the arena
boundaries, while the agent cannot exit but does not bounce (i.e. it has
inelastic collisions with the boundaries). Trials only terminate when the agent
is caught by a predator. The subject controls the agent with a joystick.
"""

import collections
import numpy as np
import os

from moog import action_spaces
from moog import observers
from moog import physics as physics_lib
from moog import shapes
from moog import tasks
from moog.state_initialization import distributions as distribs
from moog.state_initialization import sprite_generators


def get_config(num_predators):
    """Get config dictionary of kwargs for environment constructor.
    
    Args:
        num_predators: Int. Number of predators.
    """

    ############################################################################
    # Sprite initialization
    ############################################################################

    # Agent
    agent_factors = distribs.Product(
        [distribs.Continuous('x', 0., 1.),
         distribs.Continuous('y', 0., 1.)],
         shape='circle', scale=0.1, c0=0.33, c1=1., c2=0.66,
    )

    # Predators
    predator_factors = distribs.Product(
        [distribs.Continuous('x', 0., 1.),
         distribs.Continuous('y', 0., 1.)],
        shape='circle', scale=0.1, c0=0., c1=1., c2=0.8,
    )

    # Walls
    walls = shapes.border_walls(visible_thickness=0., c0=0., c1=0., c2=0.5)

    # Create callable initializer returning entire state
    agent_generator = sprite_generators.generate_sprites(
        agent_factors, num_sprites=1)
    predator_generator = sprite_generators.generate_sprites(
        predator_factors, num_sprites=num_predators)

    def state_initializer():
        agent = agent_generator(without_overlapping=walls)
        predators = predator_generator(without_overlapping=walls)
        state = collections.OrderedDict([
            ('walls', walls),
            ('agent', agent),
            ('predators', predators),
        ])
        return state

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
    # Final config
    ############################################################################

    config = {
        'state_initializer': state_initializer,
        'physics': physics,
        'task': task,
        'action_space': action_space,
        'observers': {'image': observer},
    }
    return config