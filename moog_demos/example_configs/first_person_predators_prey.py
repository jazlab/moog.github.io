"""First-person predator-prey task in an infinite plane.

In an inifite plane, predators and prey drift around and the agent is rewarded 
for catching prey and punished for being caught by predators. The field of view
travels with the agent, keeping the agent centered at all times (first-person),
and there is an occluding annulus around the agent.

The subject receives positive reward proportional to a prey's size when a prey
is caught, and negative reward proportional to the predator's size when the
agent is caught by a predator.
"""

import collections
import numpy as np

from moog import action_spaces
from moog import observers
from moog import physics as physics_lib
from moog import shapes
from moog import sprite
from moog import game_rules
from moog import tasks
from moog.state_initialization import distributions as distribs
from moog.state_initialization import sprite_generators

# Extent around the [0, 1] range that may be visible. Also used as distance from
# which predators and prey are spawned.
_FIELD_BUFFER = 0.7

# Distance beyond which predators vanish. Must be at least _FIELD_BUFFER.
_VANISH_DIST = 1.2

# Width and height of each cell in the background grid
_GRID_SIZE = 0.4


def _get_boundary_pos_distribution(boundary_buffer):
    """Get distribution generating positions from a square boundary."""
    boundary_range = [-1. * boundary_buffer, 1. + boundary_buffer]
    initialize_left = distribs.Product(
        [distribs.Continuous('y', *boundary_range)], x=boundary_range[0])
    initialize_right = distribs.Product(
        [distribs.Continuous('y', *boundary_range)], x=boundary_range[1])
    initialize_bottom = distribs.Product(
        [distribs.Continuous('x', *boundary_range)], y=boundary_range[0])
    initialize_top = distribs.Product(
        [distribs.Continuous('x', *boundary_range)], y=boundary_range[1])
    position_distribution = distribs.Mixture([
        initialize_left, initialize_right, initialize_bottom, initialize_top])
    return position_distribution


def _get_vel_distribution(min_vel, max_vel):
    """Get distribution generating velocities from a square annulus."""
    vel_distribution = distribs.SetMinus(
        distribs.Product([
            distribs.Continuous('x_vel', -1. * max_vel, max_vel),
            distribs.Continuous('y_vel', -1. * max_vel, max_vel),
        ]),
        hold_out=distribs.Product([
            distribs.Continuous('x_vel', -1. * min_vel, min_vel),
            distribs.Continuous('y_vel', -1. * min_vel, min_vel),
        ]),
    )
    return vel_distribution


def get_config(_):
    """Get environment config."""

    ############################################################################
    # Sprite initialization
    ############################################################################

    # Agent
    agent = sprite.Sprite(
        x=0.5, y=0.5, shape='circle', scale=0.04, c0=0.33, c1=1., c2=0.66)
    annulus_vertices = shapes.annulus_vertices(
        inner_radius=0.08, outer_radius=0.3)
    agent_annulus = sprite.Sprite(
        x=0.5, y=0.5, shape=annulus_vertices, scale=1., c0=0.6, c1=1., c2=1.)
    
    # Predator generator
    max_predator_vel = 0.02
    predator_pos = _get_boundary_pos_distribution(_FIELD_BUFFER)
    predator_vel = _get_vel_distribution(
        0.5 * max_predator_vel, max_predator_vel)
    predator_factors = distribs.Product(
        [predator_pos, predator_vel,
         distribs.Continuous('scale', 0.07, 0.13)],
        shape='circle', c0=0., c1=1., c2=0.8,
    )

    # Prey generator
    max_prey_vel = 0.01
    prey_pos = _get_boundary_pos_distribution(_FIELD_BUFFER)
    prey_vel = _get_vel_distribution(0.5 * max_prey_vel, max_prey_vel)
    prey_factors = distribs.Product(
        [prey_pos, prey_vel,
         distribs.Continuous('scale', 0.07, 0.13)],
        shape='circle', c0=0.2, c1=1., c2=1.,
    )

    # Grid
    grid = shapes.grid_lines(
        grid_x=_GRID_SIZE, grid_y=_GRID_SIZE, buffer_border=1., c0=0., c1=0.,
        c2=0.5)

    def state_initializer():
        state = collections.OrderedDict([
            ('grid', grid),
            ('prey', []),
            ('agent', [agent]),
            ('predators', []),
            ('agent_annulus', [agent_annulus]),
        ])
        return state

    ############################################################################
    # Physics
    ############################################################################

    agent_friction_force = physics_lib.Drag(coeff_friction=0.25)
    physics = physics_lib.Physics(
        (agent_friction_force, ['agent', 'agent_annulus']),
        updates_per_env_step=10,
    )

    ############################################################################
    # Task
    ############################################################################

    def _predator_reward_fn(_, predator_sprite):
        return -2. * predator_sprite.scale
    predator_task = tasks.ContactReward(
        reward_fn=_predator_reward_fn,
        layers_0='agent',
        layers_1='predators',
        reset_steps_after_contact=0,
    )
    def _prey_reward_fn(_, prey_sprite):
        return prey_sprite.scale
    prey_task = tasks.ContactReward(
        reward_fn=_prey_reward_fn,
        layers_0='agent',
        layers_1='prey',
    )
    task = tasks.CompositeTask(predator_task, prey_task)

    ############################################################################
    # Action space
    ############################################################################

    action_space = action_spaces.Joystick(
        scaling_factor=0.003,
        action_layers=('agent', 'agent_annulus'),
        constrained_lr=False,
    )

    ############################################################################
    # Observer
    ############################################################################

    _polygon_modifier = observers.polygon_modifiers.FirstPersonAgent(
        agent_layer='agent')
    observer = observers.PILRenderer(
        image_size=(64, 64),
        anti_aliasing=1,
        color_to_rgb='hsv_to_rgb',
        polygon_modifier=_polygon_modifier,
    )

    ############################################################################
    # Game rules
    ############################################################################

    # Make predators appear randomly
    predator_appear_generator = sprite_generators.generate_sprites(
        predator_factors, num_sprites=1)
    predator_appear = game_rules.ConditionalRule(
        condition=lambda state: np.random.binomial(1, p=0.5),
        rules=game_rules.CreateSprites('predators', predator_appear_generator),
    )

    # Make prey appear randomly
    prey_appear_generator = sprite_generators.generate_sprites(
        prey_factors, num_sprites=1)
    prey_appear = game_rules.ConditionalRule(
        condition=lambda state: np.random.binomial(1, p=0.2),
        rules=game_rules.CreateSprites('prey', prey_appear_generator),
    )

    # Make predators and prey vanish when they are distant enough and moving
    # away.
    vanish_range = [-1. * _VANISH_DIST, 1. + _VANISH_DIST]
    def _should_vanish(s):
        pos_too_small = (s.position < vanish_range[0]) * (s.velocity < 0.)
        pos_too_large = (s.position > vanish_range[1]) * (s.velocity > 0.)
        return any(pos_too_small) or any(pos_too_large)
    predator_vanish = game_rules.VanishByFilter('predators', _should_vanish)
    prey_vanish = game_rules.VanishByFilter('prey', _should_vanish)

    # Keep agent near center
    keep_near_center = game_rules.KeepNearCenter(
        agent_layer='agent',
        layers_to_center=['agent_annulus', 'predators', 'prey'],
        grid_x=_GRID_SIZE,
    )

    # Make prey vanish when contacted by agent
    prey_caught = game_rules.VanishOnContact(
        vanishing_layer='prey', contacting_layer='agent')

    rules = (predator_appear,
             prey_appear,
             prey_vanish,
             predator_vanish,
             keep_near_center,
             prey_caught)

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
