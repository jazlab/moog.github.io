"""Maze task with predators, prey, and boosters.

The predators (red circles) chase the agent. The agent receives reward for
catching prey (yellow circles), which disappear upon capture. The boosters (blue
triangles) temporarily increase the agent's speed. The portals (white squares) 
teleport the agent from one place to another.
"""

import collections
import numpy as np

from moog import action_spaces
from moog import game_rules
from moog import observers
from moog import physics as physics_lib
from moog import shapes
from moog import sprite
from moog import tasks
from moog.state_initialization import distributions as distribs
from moog.state_initialization import sprite_generators


class Booster(game_rules.AbstractRule):
    """Game rule for boosters.

    This rule makes the agent temporarily brigher in color and lighter in mass
    (to move faster) upon contact with a booster sprite.
    """

    def __init__(self,
                 mass_multiplier=0.4,
                 c2_multiplier=0.1,
                 boost_duration=60,
                 agent_layer='agent',
                 booster_layer='boosters'):
        """Constructor.

        Args:
            mass_multiplier: Float. Mass of agent after contacting a booster.
            c2_multiplier: Float. Brightness increase factor upon contacting a
                booster.
            boost_duration: Int. Number of steps before boost ends.
            agent_layer: String. Name of the agent layer in the state.
            booster_layer: String. Name of the booster layer in the state.
        """
        self._mass_multiplier = mass_multiplier
        self._c2_multiplier = c2_multiplier
        self.boost_duration = boost_duration
        self._agent_layer = agent_layer
        self._booster_layer = booster_layer

    def reset(self, state, meta_state):
        del state
        del meta_state
        self._steps_until_expire = np.inf

    def _apply_change(self, agent):
        agent.mass *= self._mass_multiplier
        agent.c2 = 1. - (1. - agent.c2) * self._c2_multiplier

    def _revert_change(self, agent):
        agent.mass /= self._mass_multiplier
        agent.c2 = 1. - (1. - agent.c2) / self._c2_multiplier

    def step(self, state, meta_state):
        del meta_state

        self._steps_until_expire -= 1
        agent = state[self._agent_layer][0]

        if self._steps_until_expire == np.inf:
            boosters = state[self._booster_layer]
            if any([agent.overlaps_sprite(s) for s in boosters]):
                self._apply_change(agent)
                self._steps_until_expire = self.boost_duration
        elif self._steps_until_expire <= 0:
            self._revert_change(agent)
            self._steps_until_expire = np.inf


def get_config(_):
    """Get environment config."""

    ############################################################################
    # Sprite initialization
    ############################################################################

    # Agent
    agent_factors = distribs.Product(
        [distribs.Continuous('x', 0.1, 0.9),
         distribs.Continuous('y', 0.1, 0.9)],
        shape='circle', scale=0.1, c0=0.33, c1=1., c2=0.7,
    )

    # Predators
    predator_factors = distribs.Product(
        [distribs.Continuous('x', 0.1, 0.9),
         distribs.Continuous('y', 0.1, 0.9)],
        shape='circle', scale=0.1, c0=0., c1=1., c2=0.8,
    )

    # Prey
    prey_factors = distribs.Product(
        [distribs.Continuous('x', 0.1, 0.9),
         distribs.Continuous('y', 0.1, 0.9)],
        shape='circle', scale=0.1, c0=0.2, c1=1., c2=1.,
    )

    # Boosters
    booster_factors = distribs.Product(
        [distribs.Continuous('x', 0.1, 0.9),
         distribs.Continuous('y', 0.1, 0.9)],
        shape='triangle', scale=0.1, c0=0.6, c1=1., c2=1.,
    )

    # Portals
    portal_factors = dict(shape='square', scale=0.1, c0=0., c1=0., c2=0.95)
    portal_sprites = [
        sprite.Sprite(x=0.125, y=0.125, **portal_factors),
        sprite.Sprite(x=0.875, y=0.875, **portal_factors),
    ]

    # Walls
    wall_color = dict(c0=0., c1=0., c2=0.5)
    island_wall_shape_0 = np.array(
        [[0.2, 0.2], [0.4, 0.2], [0.4, 0.4], [0.2, 0.4]])
    island_wall_shapes = [
        island_wall_shape_0,
        island_wall_shape_0 + np.array([[0., 0.4]]),
        island_wall_shape_0 + np.array([[0.4, 0.4]]),
        island_wall_shape_0 + np.array([[0.4, 0.]]),
    ]
    island_walls = [
        sprite.Sprite(shape=shape, x=0., y=0., **wall_color)
        for shape in island_wall_shapes
    ]
    boundary_walls = shapes.border_walls(visible_thickness=0.05, **wall_color)
    walls = boundary_walls + island_walls

    # Callable sprite generators
    agent_generator = sprite_generators.generate_sprites(
        agent_factors, num_sprites=1)
    predator_generator = sprite_generators.generate_sprites(
        predator_factors, num_sprites=1)
    prey_generator = sprite_generators.generate_sprites(
        prey_factors, num_sprites=lambda: np.random.randint(2, 5))
    booster_generator = sprite_generators.generate_sprites(
        booster_factors, num_sprites=2)

    # Create callable initializer returning entire state
    def state_initializer():
        portals = portal_sprites
        agent = agent_generator(without_overlapping=walls)
        predators = predator_generator(without_overlapping=walls + agent)
        boosters = booster_generator(without_overlapping=walls + agent)
        prey = prey_generator(without_overlapping=walls)
        state = collections.OrderedDict([
            ('walls', walls),
            ('portals', portals),
            ('boosters', boosters),
            ('prey', prey),
            ('predators', predators),
            ('agent', agent),
        ])
        return state

    ############################################################################
    # Physics
    ############################################################################

    agent_friction_force = physics_lib.Drag(coeff_friction=0.25)
    predator_friction_force = physics_lib.Drag(coeff_friction=0.05)
    predator_random_force = physics_lib.RandomForce(max_force_magnitude=0.02)
    prey_friction_force = physics_lib.Drag(coeff_friction=0.02)
    prey_random_force = physics_lib.RandomForce(max_force_magnitude=0.02)
    predator_attraction = physics_lib.DistanceForce(
        force_fn=physics_lib.linear_force_fn(zero_intercept=-0.002, slope=0.001)
    )
    asymmetric_collision = physics_lib.Collision(
        elasticity=0.25, symmetric=False, update_angle_vel=False)
    
    forces = (
        (agent_friction_force, 'agent'),
        (predator_friction_force, 'predators'),
        (predator_random_force, 'predators'),
        (prey_friction_force, 'prey'),
        (prey_random_force, 'prey'),
        (predator_attraction, 'agent', 'predators'),
        (asymmetric_collision, ['agent', 'predators', 'prey'], 'walls'),
    )
    
    physics = physics_lib.Physics(*forces, updates_per_env_step=5)

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
        predator_task, prey_task, reset_task, timeout_steps=400)

    ############################################################################
    # Action space
    ############################################################################

    action_space = action_spaces.Joystick(
        scaling_factor=0.01,
        action_layers='agent',
    )

    ############################################################################
    # Observer
    ############################################################################

    observer = observers.PILRenderer(
        image_size=(64, 64),
        anti_aliasing=1,
        color_to_rgb='hsv_to_rgb',
    )

    ############################################################################
    # Game rules
    ############################################################################

    disappear_rule = game_rules.VanishOnContact(
        vanishing_layer='prey', contacting_layer='agent')
    portal_rule = game_rules.Portal(
        teleporting_layer='agent', portal_layer='portals')
    rules = (disappear_rule, portal_rule, Booster())

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
