"""Multi-agent cleanup task.

This task is inspired by and similar to the CleanUp task introduced in this
paper:
    "Learning Reciprocity in Complex Sequential Social Dilemmas",
    Tom Eccles, Edward Hughes, János Kramár, Steven Wheelwright, & Joel Z. Leibo
    2019, arXiv, 1903.08082

The idea of this task is that there are blue fountains at the top of the arena
and green fruit at the bottom of the arena. Each fountain can be poisoned (dull
color) or clean (bright color) and each fruit can be spoiled (dull color) or
ripe (bright color). If an agent contacts a poisoned fountain, that fountain
turns clean and a spoiled fruit turns ripe. If an agent contacts a ripe fruit,
the agent receives a reward but that fruit turns spoiled and a fountain becomes
poisoned.

So the agents have to go up to the fountains, clean them up to turn the fruit
ripe, then go down to the fruit and collect them (receiving reward), then when
there's no ripe fruit left they have to go back up to the fountains.

Some agents can be selfish free-riders by hanging out near the fruit and eating
them while others to clean the fountains. Other agents can be selfless by
cleaning fountains all the time and never eating fruit.

Note: Since this is a multi-agent task, the demo cannot be run directly on it
because the joystick only controls one agent. However, see
../../../multi_agent_example/configs/cleanup.py for a config that calls this
environment and creates hand-crafted agents so the demo will work.
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

_SPRITE_SCALE = 0.1

# Value color component in HSV space corresponding to clean water and ripe fruit
_GOOD_VALUE = 1.
# Value color component in HSV space corresponding to bad water and bad fruit
_BAD_VALUE = 0.3
# Anything between _GOOD_VALUE and _BAD_VALUE
_VALUE_THRESHOLD = 0.6


def get_config(_):
    """Get environment config."""

    ############################################################################
    # Sprite initialization
    ############################################################################

    # Agents
    agent_factors = distribs.Product(
        [distribs.Continuous('x', 0., 1.),
         distribs.Continuous('y', 0.35, 0.65)],
        shape='circle', scale=0.1, c1=1., c2=0.7,
    )
    agent_0_factors = distribs.Product([agent_factors], c0=0.2)
    agent_1_factors = distribs.Product([agent_factors], c0=0.1)
    agent_2_factors = distribs.Product([agent_factors], c0=0.)

    # Walls
    walls = shapes.border_walls(visible_thickness=0.05, c0=0., c1=0., c2=0.5)

    # Fountains
    fountain_factors = {
        'shape': 'circle', 'scale': 0.05, 'c0': 0.6, 'c1': 1., 'c2': _BAD_VALUE}
    fountains_across = np.linspace(0.1, 0.9, 6)
    fountains_up = np.linspace(0.75, 0.9, 2)
    fountains_grid_x, fountains_grid_y = np.meshgrid(fountains_across,
                                                     fountains_up)
    fountains_positions = zip(np.ravel(fountains_grid_x),
                              np.ravel(fountains_grid_y))
    fountain_sprites = [
        sprite.Sprite(x=x, y=y, **fountain_factors)
        for (x, y) in fountains_positions
    ]

    # Fruits
    fruit_factors = {
        'shape': 'circle', 'scale': 0.05, 'c0': 0.3, 'c1': 1., 'c2': _BAD_VALUE}
    fruits_across = np.linspace(0.1, 0.9, 6)
    fruits_up = np.linspace(0.1, 0.25, 2)
    fruits_grid_x, fruits_grid_y = np.meshgrid(fruits_across, fruits_up)
    fruits_positions = zip(np.ravel(fruits_grid_x), np.ravel(fruits_grid_y))
    fruit_sprites = [
        sprite.Sprite(x=x, y=y, **fruit_factors)
        for (x, y) in fruits_positions
    ]

    # Create callable initializer returning entire state
    agent_0_generator = sprite_generators.generate_sprites(
        agent_0_factors, num_sprites=1)
    agent_1_generator = sprite_generators.generate_sprites(
        agent_1_factors, num_sprites=1)
    agent_2_generator = sprite_generators.generate_sprites(
        agent_2_factors, num_sprites=1)

    def state_initializer():
        agent_0 = agent_0_generator(without_overlapping=walls)
        agent_1 = agent_1_generator(without_overlapping=walls)
        agent_2 = agent_2_generator(without_overlapping=walls)
        state = collections.OrderedDict([
            ('walls', walls),
            ('fountains', fountain_sprites),
            ('fruits', fruit_sprites),
            ('agent_2', agent_2),
            ('agent_1', agent_1),
            ('agent_0', agent_0),
        ])
        return state

    ############################################################################
    # Physics
    ############################################################################

    agent_friction_force = physics_lib.Drag(coeff_friction=0.25)
    asymmetric_collision = physics_lib.Collision(
        elasticity=0.25, symmetric=False)
    
    forces = (
        (agent_friction_force, ['agent_0', 'agent_1', 'agent_2']),
        (asymmetric_collision, ['agent_0', 'agent_1', 'agent_2'], 'walls'),
    )
    
    physics = physics_lib.Physics(*forces, updates_per_env_step=5)

    ############################################################################
    # Task
    ############################################################################

    task = tasks.ContactReward(
        1, layers_0='agent_0', layers_1='fruits',
        condition=lambda s_0, s_1: s_1.c2 > _VALUE_THRESHOLD,
    )

    ############################################################################
    # Action space
    ############################################################################

    action_space = action_spaces.Composite(
        agent_0=action_spaces.Joystick(
            scaling_factor=0.005, action_layers='agent_0'),
        agent_1=action_spaces.Joystick(
            scaling_factor=0.005, action_layers='agent_1'),
        agent_2=action_spaces.Joystick(
            scaling_factor=0.005, action_layers='agent_2'),
    )

    ############################################################################
    # Observer
    ############################################################################

    image_observer = observers.PILRenderer(
        image_size=(64, 64),
        anti_aliasing=1,
        color_to_rgb='hsv_to_rgb',
    )
    raw_state_observer = observers.RawState()  # needed by hand-crafted agents

    ############################################################################
    # Game rules
    ############################################################################

    def _spoil_fruit(sprite):
        sprite.c2 = _BAD_VALUE
    def _ripen_fruit(sprite):
        sprite.c2 = _GOOD_VALUE
    def _poison_fountain(sprite):
        sprite.c2 = _BAD_VALUE
    def _clean_fountain(sprite):
        sprite.c2 = _GOOD_VALUE

    def agents_contacting_layer(state, layer, value):
        n_contact = 0
        for s in state[layer]:
            if s.c2 != value:
                continue
            n_contact += (
                s.overlaps_sprite(state['agent_0'][0]) or 
                s.overlaps_sprite(state['agent_1'][0]) or 
                s.overlaps_sprite(state['agent_2'][0])
            )
        return n_contact
    
    poison_fountains = game_rules.ModifySprites(
        layers='fountains', modifier=_poison_fountain, sample_one=True,
        filter_fn=lambda s: s.c2 > _VALUE_THRESHOLD)
    poison_fountains = game_rules.ConditionalRule(
        condition=lambda s: agents_contacting_layer(s, 'fruits', _GOOD_VALUE),
        rules=poison_fountains,
    )
    ripen_fruits = game_rules.ModifySprites(
        layers='fruits', modifier=_ripen_fruit, sample_one=True,
        filter_fn=lambda s: s.c2 < _VALUE_THRESHOLD)
    ripen_fruits = game_rules.ConditionalRule(
        condition=lambda s: agents_contacting_layer(s, 'fountains', _BAD_VALUE),
        rules=ripen_fruits,
    )

    spoil_fruits = game_rules.ModifyOnContact(
        layers_0='fruits',
        layers_1=('agent_0', 'agent_1', 'agent_2'),
        modifier_0=_spoil_fruit,
        filter_0=lambda s: s.c2 > _VALUE_THRESHOLD)
    clean_fountains = game_rules.ModifyOnContact(
        layers_0='fountains',
        layers_1=('agent_0', 'agent_1', 'agent_2'),
        modifier_0=_clean_fountain,
        filter_0=lambda s: s.c2 < _VALUE_THRESHOLD)
    
    rules = (poison_fountains, spoil_fruits, ripen_fruits, clean_fountains)

    ############################################################################
    # Final config
    ############################################################################

    config = {
        'state_initializer': state_initializer,
        'physics': physics,
        'task': task,
        'action_space': action_space,
        'observers': {'image': image_observer, 'state': raw_state_observer},
        'game_rules': rules,
    }
    return config
