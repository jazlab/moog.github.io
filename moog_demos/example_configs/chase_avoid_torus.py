"""Chase/avoid task on a torus.

In this task there are predators (red circles), prey (yellow circles) and an
agent (green square). The subject controls the agent with a joystick. The
subject's goal is to catch the prey while avoiding being caught by the
predators. The prey are repulsed by the agent and predators are attracted to the
agent. Prey and predators move stochasticity and with constant speed.

The environment geometry is shaped like a torus --- when an object reaches one
boundary, it reappears on the opposite boundary.
"""

import collections
import itertools
import numpy as np

from moog import action_spaces
from moog import game_rules
from moog import observers
from moog import physics as physics_lib
from moog import sprite
from moog import tasks
from moog.state_initialization import distributions as distribs
from moog.state_initialization import sprite_generators
from moog.observers import polygon_modifiers


def _get_config(num_prey, num_predators):
    """Get environment config."""

    ############################################################################
    # Sprite initialization
    ############################################################################

    # Agent
    agent_factors = distribs.Product(
        [distribs.Continuous('x', 0., 1.),
         distribs.Continuous('y', 0., 1.)],
        scale=0.08, c0=0, c1=255, c2=0,
    )

    # Predators
    predator_factors = distribs.Product(
        [distribs.Continuous('x', 0., 1.),
         distribs.Continuous('y', 0., 1.),
         distribs.Continuous('x_vel', -0.02, 0.02),
         distribs.Continuous('y_vel', -0.02, 0.02),],
        scale=0.08, shape='circle', opacity=192, c0=255, c1=0, c2=0,
    )

    # Prey
    prey_factors = distribs.Product(
        [distribs.Continuous('x', 0., 1.),
         distribs.Continuous('y', 0., 1.),
         distribs.Continuous('x_vel', -0.02, 0.02),
         distribs.Continuous('y_vel', -0.02, 0.02),],
        scale=0.08, shape='circle', opacity=192, c0=255, c1=255, c2=0,
    )

    # Create callable initializer returning entire state
    predator_generator = sprite_generators.generate_sprites(
        predator_factors, num_sprites=num_predators)
    prey_generator = sprite_generators.generate_sprites(
        prey_factors, num_sprites=num_prey)

    def state_initializer():
        """Callable returning state at every episode reset."""
        agent = sprite.Sprite(**agent_factors.sample())
        predators = predator_generator(without_overlapping=(agent,))
        prey = prey_generator(without_overlapping=(agent,))

        state = collections.OrderedDict([
            ('prey', prey),
            ('predators', predators),
            ('agent', [agent]),
        ])
        return state

    ############################################################################
    # Physics
    ############################################################################

    agent_friction_force = physics_lib.Drag(coeff_friction=0.25)
    random_force = physics_lib.RandomForce(max_force_magnitude=0.01)
    predator_attraction = physics_lib.DistanceForce(
        physics_lib.linear_force_fn(zero_intercept=-0.001, slope=0.0005))
    prey_avoid = physics_lib.DistanceForce(
        physics_lib.linear_force_fn(zero_intercept=0.001, slope=-0.0005))

    forces = (
        (agent_friction_force, 'agent'),
        (random_force, ['predators', 'prey']),
        (predator_attraction, 'agent', 'predators'),
        (prey_avoid, 'agent', 'prey'),
    )

    constant_speed = physics_lib.ConstantSpeed(
        layer_names=['prey', 'predators'], speed=0.015)

    physics = physics_lib.Physics(
        *forces,
        updates_per_env_step=10,
        corrective_physics=[constant_speed],
    )

    ############################################################################
    # Task
    ############################################################################

    predator_task = tasks.ContactReward(
        -5, layers_0='agent', layers_1='predators', reset_steps_after_contact=0)
    prey_task = tasks.ContactReward(1, layers_0='agent', layers_1='prey')
    reset_task = tasks.Reset(
        condition=lambda state: len(state['prey']) == 0,
        steps_after_condition=5,
    )
    task = tasks.CompositeTask(
        reset_task, predator_task, prey_task, timeout_steps=300)

    ############################################################################
    # Action space
    ############################################################################

    action_space = action_spaces.Joystick(
        scaling_factor=0.025, action_layers='agent', control_velocity=True)

    ############################################################################
    # Observer
    ############################################################################

    observer = observers.PILRenderer(
        image_size=(64, 64),
        anti_aliasing=1,
        polygon_modifier=polygon_modifiers.TorusGeometry(
            ['agent', 'predators', 'prey']),
    )

    ############################################################################
    # Game rules
    ############################################################################

    prey_vanish = game_rules.VanishOnContact(
        vanishing_layer='prey', contacting_layer='agent')
    def _torus_position_wrap(s):
        s.position = np.remainder(s.position, 1)
    torus_position_wrap = game_rules.ModifySprites(
        ('agent', 'predators', 'prey'), _torus_position_wrap)

    rules = (prey_vanish, torus_position_wrap)

    ############################################################################
    # Final config
    ############################################################################

    config = {
        'state_initializer': state_initializer,
        'physics': physics,
        'task': task,
        'action_space': action_space,
        'observers': {'image': observer},
        'game_rules': rules,
    }
    return config


def get_config(level):
    if level == 0:
        return _get_config(
            num_prey=1,
            num_predators=2,
        )
    elif level == 1:
        return _get_config(
            num_prey=lambda: np.random.randint(1, 3),
            num_predators=lambda: np.random.randint(1, 3),
        )
    else:
        raise ValueError('Invalid level {}'.format(level))
