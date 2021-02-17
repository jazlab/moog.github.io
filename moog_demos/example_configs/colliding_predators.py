"""Avoid colliding predator polygons.

This task serves to showcase collisions. The predators have a variety of
polygonal shapes and bounce off each other and off the walls with Newtonian
collisions. The subject controls a green agent circle. The subject gets negative
reward if contacted by a predators and positive reward periodically.
"""

import collections
import numpy as np

from moog import action_spaces
from moog import physics as physics_lib
from moog import observers
from moog import sprite
from moog import tasks
from moog import shapes
from moog.state_initialization import distributions as distribs
from moog.state_initialization import sprite_generators


def get_config(_):
    """Get environment config."""

    ############################################################################
    # Sprite initialization
    ############################################################################

    # Agent
    agent_factors = distribs.Product(
        [distribs.Continuous('x', 0.1, 0.9),
         distribs.Continuous('y', 0.1, 0.9)],
        shape='circle', scale=0.1, c0=0.33, c1=1., c2=0.66,
    )

    # Predators
    shape_0 = 1.8 * np.array(
        [[-0.3, -0.3], [0.1, -0.7], [0.4, 0.6], [-0.1, 0.25]])
    shape_1 = 1.5 * np.array(
        [[-0.5, -0.3], [-0.1, -0.7], [0.7, 0.1], [0., -0.1], [-0.3, 0.25]])
    predator_factors = distribs.Product(
        [distribs.Continuous('x', 0.2, 0.8),
         distribs.Continuous('y', 0.2, 0.8),
         distribs.Discrete(
             'shape', [shape_0, shape_1, 'star_5', 'triangle', 'spoke_5']),
         distribs.Continuous('angle', 0., 2 * np.pi),
         distribs.Continuous('aspect_ratio', 0.75, 1.25),
         distribs.Continuous('scale', 0.1, 0.15),
         distribs.Continuous('x_vel', -0.03, 0.03),
         distribs.Continuous('y_vel', -0.03, 0.03),
         distribs.Continuous('angle_vel', -0.05, 0.05)],
        c0=0., c1=1., c2=0.8,
    )

    # Walls
    walls = shapes.border_walls(visible_thickness=0.05, c0=0., c1=0., c2=0.5)

    # Create callable initializer returning entire state
    agent_generator = sprite_generators.generate_sprites(
        agent_factors, num_sprites=1)
    predator_generator = sprite_generators.generate_sprites(
        predator_factors, num_sprites=5)

    def state_initializer():
        predators = predator_generator(
            disjoint=True, without_overlapping=walls)
        agent = agent_generator(without_overlapping=walls + predators)
        state = collections.OrderedDict([
            ('walls', walls),
            ('predators', predators),
            ('agent', agent),
        ])
        return state

    ############################################################################
    # Physics
    ############################################################################

    agent_friction_force = physics_lib.Drag(coeff_friction=0.25)
    asymmetric_collision = physics_lib.Collision(
        elasticity=1., symmetric=False, update_angle_vel=True)
    symmetric_collision = physics_lib.Collision(
        elasticity=1., symmetric=True, update_angle_vel=True)
    agent_wall_collision = physics_lib.Collision(
        elasticity=0., symmetric=False, update_angle_vel=False)
    
    forces = (
        (agent_friction_force, 'agent'),
        (symmetric_collision, 'predators', 'predators'),
        (asymmetric_collision, 'predators', 'walls'),
        (agent_wall_collision, 'agent', 'walls'),
    )
    
    physics = physics_lib.Physics(*forces, updates_per_env_step=10)

    ############################################################################
    # Task
    ############################################################################

    predator_task = tasks.ContactReward(
        -5, layers_0='agent', layers_1='predators')
    stay_alive_task = tasks.StayAlive(
        reward_period=20,
        reward_value=0.2,
    )
    task = tasks.CompositeTask(
        predator_task, stay_alive_task, timeout_steps=200)

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
