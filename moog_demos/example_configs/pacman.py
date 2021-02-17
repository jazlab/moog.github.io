"""PacMan-like task.

In this task all objects move in a maze with the same speed. The maze is
randomized every trial. The subject controls a green agent. Red ghost agents
wander the maze randomly without backtracking. The subject's goal is to collect
all yellow pellets in the maze. Ghosts only begin moving once agent moves.
"""

import collections
import numpy as np

from moog import action_spaces
from moog import game_rules
from moog import maze_lib
from moog import observers
from moog import physics as physics_lib
from moog import sprite
from moog import tasks


def _get_config(num_ghosts, maze_size):
    """Get environment config."""

    ############################################################################
    # Sprite initialization
    ############################################################################

    # Agent
    agent_factors = dict(shape='circle', scale=0.05, c0=0.33, c1=1., c2=0.66)

    # Prey
    prey_factors = dict(shape='circle', scale=0.025, c0=0.2, c1=1., c2=1.)

    # Ghosts
    ghost_factors = dict(
        shape='circle', scale=0.05, mass=np.inf, c0=0., c1=1., c2=0.8)

    def state_initializer():
        maze = maze_lib.generate_random_maze_matrix(
            size=maze_size, ambient_size=12)
        maze = maze_lib.Maze(np.flip(maze, axis=0))
        walls = maze.to_sprites(c0=0., c1=0., c2=0.8)

        # Sample positions in maze grid of agent and ghosts
        n_ghosts = num_ghosts()
        points = maze.sample_distinct_open_points(1 + n_ghosts)
        positions = [maze.grid_side * (0.5 + np.array(x)) for x in points]

        # Agent
        agent_position = positions[0]
        agent = [sprite.Sprite(
            x=agent_position[1], y=agent_position[0], **agent_factors)]

        # ghosts
        ghosts = []
        for i in range(n_ghosts):
            position = positions[i + 1]
            ghosts.append(sprite.Sprite(
                x=position[1], y=position[0], **ghost_factors))

        # Place prey at every open maze location
        prey = []
        open_maze_points = np.argwhere(maze.maze == 0)
        for p in open_maze_points:
            pos = maze.grid_side * (0.5 + np.array(p))
            prey.append(sprite.Sprite(x=pos[1], y=pos[0], **prey_factors))

        state = collections.OrderedDict([
            ('walls', walls),
            ('prey', prey),
            ('ghosts', ghosts),
            ('agent', agent),
        ])
        return state

    ############################################################################
    # Physics
    ############################################################################

    maze_physics = physics_lib.MazePhysics(
        maze_layer='walls',
        avatar_layers=('agent', 'prey', 'ghosts'), constant_speed=0.015,
    )

    physics = physics_lib.Physics(
        (physics_lib.RandomMazeWalk(speed=0.015), ['ghosts']),
        updates_per_env_step=1, corrective_physics=[maze_physics],
    )

    ############################################################################
    # Task
    ############################################################################

    ghost_task = tasks.ContactReward(
        -5, layers_0='agent', layers_1='ghosts', reset_steps_after_contact=0)
    prey_task = tasks.ContactReward(1, layers_0='agent', layers_1='prey')
    reset_task = tasks.Reset(
        condition=lambda state: len(state['prey']) == 0,
        steps_after_condition=5,
    )
    task = tasks.CompositeTask(
        ghost_task, prey_task, reset_task, timeout_steps=1000)

    ############################################################################
    # Action space
    ############################################################################

    action_space = action_spaces.Grid(
        scaling_factor=0.015,
        action_layers='agent',
        control_velocity=True,
        momentum=0.5,  # Value irrelevant, since maze_physics has constant speed
    )

    ############################################################################
    # Observer
    ############################################################################

    observer = observers.PILRenderer(
        image_size=(256, 256),
        anti_aliasing=1,
        color_to_rgb='hsv_to_rgb',
    )

    ############################################################################
    # Game rules
    ############################################################################

    def _unglue(s):
        s.mass = 1.
    def _unglue_condition(state):
        return not np.all(state['agent'][0].velocity == 0)
    unglue = game_rules.ConditionalRule(
        condition=_unglue_condition,
        rules=game_rules.ModifySprites(('prey', 'ghosts'), _unglue),
    )

    vanish_on_contact = game_rules.VanishOnContact(
        vanishing_layer='prey', contacting_layer='agent')

    rules = (vanish_on_contact, unglue)

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
    """Get config dictionary of kwargs for environment constructor.
    
    Args:
        level: Int. Different values yield different maze sizes and numbers of
            ghosts.
    """
    if level == 0:
        return _get_config(
            num_ghosts=lambda: 2,
            maze_size=8,
        )
    elif level == 1:
        return _get_config(
            num_ghosts=lambda: 3,
            maze_size=10,
        )
    else:
        raise ValueError('Invalid level {}'.format(level))
