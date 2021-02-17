"""Falling balls Task.

This task is not really a task, but instead used to demonstrate errors in
collision physics.

In this task some circles fall from the top of the screen, acted on by gravity.
The collide with each other, settling onto a wall at the bottom of the screen.
There is a divider in the bottom wall to add more collisions. The balls collide
with some elasticity with the walls and each other.

The collision dynamics for this task often do not look realistic. This is
because the collision force can be unrealistic under two conditions (both of
which are present in this task):
    * When an object collides while also accelerating due to some force at a
        distance.
    * When an object undergoes multiple collisions simultaneously.

The issues arise most often with sprites that have many vertices, such as the 
circles in this task.

Consider this task a word of caution when using collisions with many moving
objects and non-collision forces.
"""

import collections
import numpy as np

from moog import action_spaces
from moog import observers
from moog import physics as physics_lib
from moog import sprite as sprite_lib
from moog import tasks
from moog.state_initialization import distributions as distribs
from moog.state_initialization import sprite_generators


def get_config(_):
    """Get environment config"""

    ############################################################################
    # Sprite initialization
    ############################################################################

    # Ball generator
    ball_factors = distribs.Product(
        [distribs.Continuous('x', 0.25, 0.75),
         distribs.Continuous('y', 0.5, 0.9),
         distribs.Continuous('x_vel', -0.01, 0.01)],
        scale=0.1, shape='circle', c0=0, c1=0, c2=255, mass=1.,
    )
    ball_generator = sprite_generators.generate_sprites(
        ball_factors, num_sprites=4)

    # Walls
    bottom_wall = [[-1, 0.1], [2, 0.1], [2, -1], [-1, -1]]
    left_wall = [[0.05, -0.1], [0.05, 1.1], [-1, 1.1], [-1, -0.1]]
    right_wall = [[0.95, -0.1], [0.95, 1.1], [2, 1.1], [2, -0.1]]
    divider = [[0.45, -1], [0.45, 0.3], [0.55, 0.3], [0.55, -1]]
    walls = [
        sprite_lib.Sprite(shape=np.array(v), x=0, y=0, c0=128, c1=128, c2=128)
        for v in [bottom_wall, left_wall, right_wall, divider]
    ]

    def state_initializer():
        """Callable returning new state at each episode reset."""
        state = collections.OrderedDict([
            ('walls', walls),
            ('balls', ball_generator(disjoint=True)),
            ('agent', []),
        ])

        return state

    ############################################################################
    # Physics
    ############################################################################

    # Setting max_recursion_depth > 0 can increase stability
    # Setting update_angle_vel = False is recommended for stability
    collision = physics_lib.Collision(
        elasticity=0.6,
        symmetric=False,
        update_angle_vel=False,
        max_recursion_depth=2,
    )
    physics = physics_lib.Physics(
        (collision, 'balls', ['balls', 'walls']),
        (physics_lib.DownGravity(g=-0.001), 'balls'),
        updates_per_env_step=20,
    )
    
    ############################################################################
    # Task
    ############################################################################

    task = tasks.CompositeTask(timeout_steps=100)

    ############################################################################
    # Action space
    ############################################################################

    # Need an action space, so let it control an empty agent layer
    action_space = action_spaces.Grid(action_layers='agent')

    ############################################################################
    # Observer
    ############################################################################

    observer = observers.PILRenderer(image_size=(64, 64), anti_aliasing=1)

    ############################################################################
    # Final config
    ############################################################################

    config = {
        'state_initializer': state_initializer,
        'physics': physics,
        'task': task,
        'action_space': action_space,
        'observers': {'image': observer},
        'game_rules': (),
    }
    return config
