"""Contact prediction task.

In this task two red balls fall into a box. They bounce elastically off the
walls of the box, ultimately disappearing off the top of the screen because
there is no gravity. The subject's goal is to predict whether they will contact
each other. There is an occluder covering the bottom portion of the box. The
occluder may be translucent, depending on the argument to get_config(_).

The main entry point is the get_config() function at the bottom of this file.
"""

import collections
import numpy as np

from moog import action_spaces
from moog import game_rules
from moog import observers
from moog import physics as physics_lib
from moog import sprite
from moog import tasks
from moog.state_initialization import distributions as distribs


def _get_config(translucent_occluder):
    """Get environment config."""

    ############################################################################
    # Physics
    ############################################################################

    elastic_collision = physics_lib.Collision(
        elasticity=1., symmetric=False, update_angle_vel=False)
    physics = physics_lib.Physics(
        (elastic_collision, 'targets', 'walls'),
        updates_per_env_step=10,
    )

    def _predict_contact(state):
        """Predict whether targets will contact."""
        while True:
            if state['targets'][0].overlaps_sprite(state['targets'][1]):
                return True
            if all(s.y > 1.1 and s.y_vel > 0 for s in state['targets']):
                # Both targets above screen and moving up
                break
            physics.step(state)
        return False

    ############################################################################
    # Sprite initialization
    ############################################################################

    # Targets
    target_y_speed = 0.02
    target_factors = distribs.Product(
        [distribs.Continuous('x', 0.15, 0.85),
         distribs.Continuous('x_vel', -target_y_speed, target_y_speed)],
        y_vel=-target_y_speed, scale=0.16, shape='circle', opacity=192, c0=255,
        c1=0, c2=0,
    )

    # Occluder
    occluder = sprite.Sprite(
        x=0.5, y=0.2, shape='square', scale=1., c0=192, c1=192, c2=128,
        opacity=128 if translucent_occluder else 255
    )

    # Walls
    bottom_wall = [[-1, 0.1], [2, 0.1], [2, -1], [-1, -1]]
    left_wall = [[0.05, -1], [0.05, 4], [-1, 4], [-1, -1]]
    right_wall = [[0.95, -1], [0.95, 4], [2, 4], [2, -1]]
    walls = [
        sprite.Sprite(shape=np.array(v), x=0, y=0, c0=128, c1=128, c2=128)
        for v in [bottom_wall, left_wall, right_wall]
    ]
    
    # Make response boxes and tokens
    response_box_factors = dict(
        y=0.05, scale=0.12, shape='square', aspect_ratio=0.5, c0=0, c1=0, c2=0)
    response_boxes = [
        sprite.Sprite(x=0.4, **response_box_factors),
        sprite.Sprite(x=0.6, **response_box_factors),
    ]
    response_token_factors = dict(
        y=0.05, scale=0.03, shape='circle', c0=255, c1=0, c2=0, opacity=192)
    response_tokens = [
        sprite.Sprite(x=x, **response_token_factors)
        for x in [0.37, 0.43, 0.59, 0.61]
    ]
    
    def state_initializer():
        """Callable returning state ordereddict each episode reset."""
        agent = sprite.Sprite(
            x=0.5, y=0.05, scale=0.03, shape='spoke_4', c0=255, c1=255, c2=255)
        target_0 = sprite.Sprite(y=1.4, **target_factors.sample())
        target_1 = sprite.Sprite(
            y=np.random.uniform(1.7, 2.4), **target_factors.sample())
        screen = sprite.Sprite(
            x=0.5, y=0.5, shape='square', c0=128, c1=128, c2=128)

        state = collections.OrderedDict([
            ('targets', [target_0, target_1]),
            ('occluders', [occluder]),
            ('walls', walls),
            ('response_boxes', response_boxes),
            ('response_tokens', response_tokens),
            ('agent', [agent]),
            ('screen', [screen]),
        ])

        # Predict whether targets will contact, putting this information in
        # agent metadata
        orig_pos = [np.copy(s.position) for s in state['targets']]
        orig_vel = [np.copy(s.velocity) for s in state['targets']]
        agent.metadata = {'will_contact': _predict_contact(state)}
        for s, pos, vel in zip(state['targets'], orig_pos, orig_vel):
            s.position = pos
            s.velocity = vel

        return state

    ############################################################################
    # Task
    ############################################################################

    def _reward_fn(state):
        agent = state['agent'][0]
        if agent.overlaps_sprite(state['response_boxes'][0]):
            # Collision response
            return -1 if agent.metadata['will_contact'] else 1
        elif agent.overlaps_sprite(state['response_boxes'][1]):
            # No collision response
            return 1 if agent.metadata['will_contact'] else -1
        else:
            return 0
            
    conditional_task = tasks.Reset(
        condition=lambda state: _reward_fn(state) != 0,
        reward_fn=_reward_fn,
        steps_after_condition=5,
    )
    task = tasks.CompositeTask(conditional_task, timeout_steps=1000)

    ############################################################################
    # Action space
    ############################################################################

    action_space = action_spaces.Grid(
        scaling_factor=0.015,
        action_layers='agent',
        control_velocity=True,
    )

    ############################################################################
    # Observer
    ############################################################################

    observer = observers.PILRenderer(image_size=(64, 64), anti_aliasing=1)

    ############################################################################
    # Game rules
    ############################################################################

    screen_vanish = game_rules.VanishByFilter('screen')
    screen_vanish = game_rules.TimedRule(
        step_interval=(15, 16), rules=(screen_vanish,))

    rules = (screen_vanish,)

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


def get_config(translucent_occluder):
    return _get_config(translucent_occluder)
