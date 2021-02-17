"""Red-Green Task.

This task is a variant of the task used in this paper:
Smith, K. A., Peres, F., Vul, E., & Tenebaum, J. (2017). Thinking inside the
box: Motion prediction in contained spaces uses simulation. In CogSci.

In this task, there is a blue ball that bounces in an enclosed rectangular
arena. The arena may have gray rectangular obstacles that the blue ball bounces
off. The arena has one green box and one red box. The subject's goal is to
predict which of the green/red boxes the blue ball will contact first.

In this particular implementation, the subject moves a token at the bottom of
the screen left or right to indicate its choice.

The main entrypoint is the get_config() function at the bottom of this file.
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
from moog.state_initialization import sprite_generators


class RadialVelocity(distribs.AbstractDistribution):
    """Radial velocity distribution."""

    def __init__(self, speed):
        """Constructor.

        Args:
            speed: Float. Speed of the sampled velocities.
        """
        self._speed = speed

    def sample(self, rng):
        rng = self._get_rng(rng)
        theta = rng.uniform(0., 2 * np.pi)
        x_vel = self._speed * np.cos(theta)
        y_vel = self._speed * np.sin(theta)
        return {'x_vel': x_vel, 'y_vel': y_vel}

    def contains(self, spec):
        if 'x_vel' not in spec or 'y_vel' not in spec:
            return False

        vel_norm = np.linalg.norm([spec['x_vel'], spec['y_vel']])
        if np.abs(vel_norm - self._speed) < _EPSILON:
            return True
        else:
            return False

    def to_str(self, indent):
        s = 'RadialVelocity({})'.format(self._speed)
        return indent * '  ' + s

    @property
    def keys(self):
        return set(['x_vel', 'y_vel'])
        

def _get_config(num_obstacles, valid_step_range):
    """Get environment config.
    
    Args:
        num_obstacles: Int. Number of obstacles.
        valid_step_range: 2-iterable of ints. (min_num_steps, max_num_steps).
            All trials must have duration in this step range.
    
    Returns:
        config: Config dictionary to pass to environment constructor.
    """

    ############################################################################
    # Physics
    ############################################################################

    elastic_collision = physics_lib.Collision(
        elasticity=1., symmetric=False, update_angle_vel=False)
    physics = physics_lib.Physics(
        (elastic_collision, 'ball', 'walls'),
        updates_per_env_step=10,
    )

    def _predict_trial_end(state):
        """Predict whether a trial will end in step range and true response.

        Args:
            state: OrderedDict of sprite layers. Initial state of environment.
        
        Returns:
            valid_trial: Bool. Whether trial will end with number of steps in
                valid_step_range.
            contact_color: Binary. 0 if ball will contact red first, 1 if it
                will contact green first.
        """
        for step in range(valid_step_range[1]):
            red_overlap = state['ball'][0].overlaps_sprite(state['red'][0])
            green_overlap = state['ball'][0].overlaps_sprite(state['green'][0])
            if red_overlap or green_overlap:
                if step < valid_step_range[0]:
                    return False, None
                else:
                    contact_color = 0 if red_overlap else 1
                    return True, contact_color
            physics.step(state)
        return False, None
    
    ############################################################################
    # Sprite initialization
    ############################################################################

    # Ball generator
    ball_factors = distribs.Product(
        [distribs.Continuous('x', 0.15, 0.85),
         distribs.Continuous('y', 0.15, 0.85),
         RadialVelocity(speed=0.03)],
        scale=0.05, shape='circle', c0=64, c1=64, c2=255,
    )
    ball_generator = sprite_generators.generate_sprites(
        ball_factors, num_sprites=1, max_recursion_depth=100,
        fail_gracefully=True)

    # Obstacle generator
    obstacle_factors = distribs.Product(
        [distribs.Continuous('x', 0.2, 0.8),
         distribs.Continuous('y', 0.2, 0.8)],
        scale=0.2, shape='square', c0=128, c1=128, c2=128,
    )
    obstacle_generator = sprite_generators.generate_sprites(
        obstacle_factors, num_sprites=2 + num_obstacles,
        max_recursion_depth=100, fail_gracefully=True)

    # Walls
    bottom_wall = [[-1, 0.1], [2, 0.1], [2, -1], [-1, -1]]
    top_wall = [[-1, 0.95], [2, 0.95], [2, 2], [-1, 2]]
    left_wall = [[0.05, -1], [0.05, 4], [-1, 4], [-1, -1]]
    right_wall = [[0.95, -1], [0.95, 4], [2, 4], [2, -1]]
    walls = [
        sprite.Sprite(shape=np.array(v), x=0, y=0, c0=128, c1=128, c2=128)
        for v in [bottom_wall, top_wall, left_wall, right_wall]
    ]

    def state_initializer():
        """Callable returning new state at each episode reset."""
        obstacles = obstacle_generator(disjoint=True)
        ball = ball_generator(without_overlapping=obstacles)
        if len(obstacles) < num_obstacles + 2 or not ball:
            # Max recursion depth failed trying to generate without overlapping
            return state_initializer()
        
        red = obstacles[0]
        green = obstacles[1]
        obstacles = obstacles[2:]

        # Set the colors of the red and green boxes
        red.c0 = 255
        red.c1 = 0
        red.c2 = 0
        green.c0 = 0
        green.c1 = 255
        green.c2 = 0

        # Create agent and response tokens at the bottom of the sreen
        agent = sprite.Sprite(x=0.5, y=0.06, shape='spoke_4', scale=0.03, c0=255, c1=255, c2=255)
        responses = [
            sprite.Sprite(x=0.6, y=0.06, shape='square', scale=0.03, c0=255, c1=0, c2=0),
            sprite.Sprite(x=0.4, y=0.06, shape='square', scale=0.03, c0=0, c1=255, c2=0),
        ]

        state = collections.OrderedDict([
            ('walls', walls + obstacles),
            ('red', [red]),
            ('green', [green]),
            ('ball', ball),
            ('responses', responses),
            ('agent', [agent]),
        ])

        # Rejection sampling if trial won't finish in valid step range
        original_ball_position = np.copy(ball[0].position)
        original_ball_velocity = np.copy(ball[0].velocity)
        valid_trial, contact_red = _predict_trial_end(state)
        if valid_trial:
            ball[0].position = original_ball_position
            ball[0].velocity = original_ball_velocity
            agent.metadata = {'true_contact_color': contact_red}
        else:
            return state_initializer()

        return state

    ############################################################################
    # Task
    ############################################################################

    def _reward_fn(sprite_agent, sprite_response):
        response_green = sprite_response.c0 < 128
        if sprite_agent.metadata['true_contact_color'] == response_green:
            return 1.
        else:
            return -1.
            
    contact_reward = tasks.ContactReward(
        reward_fn=_reward_fn,
        layers_0='agent',
        layers_1='responses',
        reset_steps_after_contact=10,
    )
    task = tasks.CompositeTask(contact_reward, timeout_steps=400)

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

    # Stop ball on contact with red or green box
    def _stop_ball(s):
        s.velocity = np.zeros(2)
    stop_ball = game_rules.ModifyOnContact(
        layers_0='ball',
        layers_1=('red', 'green'),
        modifier_0=_stop_ball
    )

    ############################################################################
    # Final config
    ############################################################################

    config = {
        'state_initializer': state_initializer,
        'physics': physics,
        'task': task,
        'action_space': action_space,
        'observers': {'image': observer},
        'game_rules': (stop_ball,),
    }
    return config


def get_config(level):
    """Get config to pass to environment constructor.

    Args:
        level: Int. Number of obstacles in arena.
    """
    if not isinstance(level, int):
        raise ValueError(f'level is {level}, but must be an integer.')
    return _get_config(num_obstacles=level, valid_step_range=(50, 150))
