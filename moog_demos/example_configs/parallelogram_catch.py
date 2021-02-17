"""First-person shape catcher task with parallelogram arrangement of prey.

In this task the subject controls a green circlular agent with a joystick. The
motion is first-person, so the agent is fixed at the center of the screen while
everything else moves. There are four yellow prey sprites. These prey sprites
are identical parallelograms, and they are spatially arranged in a parallelogram
with the same aspect ratio. There is an annulus occluding all peripheral vision,
i.e. never are two prey visible simultaneously.

The entire prey configuration may be drifting and rotating, depending on the
level. See the get_config() function at the bottom of this file.

This forces the subject to make a hierarchical inference task. After seeing the
first prey, there are four possible arrangements of the other prey. After
finding the second prey, there are two possible arrangements of the remaining
prey. After finding the third prey, the fourth prey's position is deterministic.
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

# Width and height of each cell in the background grid
_GRID_SIZE = 0.3


def get_parallelogram(min_axis_ratio=0.4):
    """Get parallelogram vertices centered around 0 with maximum radius 1."""
    angles = np.pi * (np.array([0., 0.5, 1., 1.5]) + np.random.uniform(0, 2))
    vertices = np.stack((np.sin(angles), np.cos(angles)), axis=1)
    axis_ratio = np.random.uniform(min_axis_ratio, 1.)
    vertices *= np.array([[1.], [axis_ratio], [1.], [axis_ratio]])

    return vertices


def get_prey(centered_vertices, scale=1., max_vel=0., sprite_scale=0.1):
    """Get prey sprites.

    Args:
        centered_vertices: Numpy array of shape [num_vertices, 2] containing
            vertex positions.
        scale: Re-scaling factor of centered_vertices for the global space.
        max_vel: Maximum velocity of the sprites.
        sprite_scale: Re-scaling factor of centered_vertices for the individual
            sprite shapes.
    """
    sprite_shape = sprite_scale * centered_vertices
    sprite_positions = scale * centered_vertices
    sprite_positions += np.array([0.5, 0.5]) - sprite_positions[0]

    # We sample each sprite's velocity independently so that the entire tethered
    # configuration may rotate
    prey = [
        sprite.Sprite(
            x=pos[0], y=pos[1], shape=sprite_shape, scale=1.,
            x_vel=np.random.uniform(-1 * max_vel, max_vel),
            y_vel=np.random.uniform(-1 * max_vel, max_vel),
            c0=0.2, c1=1., c2=1.)
        for pos in sprite_positions
    ]
    return prey


def _get_config(max_vel):
    """Get environment config."""

    ############################################################################
    # Sprite initialization
    ############################################################################

    # Grid
    grid = shapes.grid_lines(
        grid_x=_GRID_SIZE, grid_y=_GRID_SIZE, buffer_border=1., c0=0., c1=0.,
        c2=0.5)

    def state_initializer():
        agent = sprite.Sprite(
            x=0.5, y=0.5, shape='circle', scale=0.04, c0=0.33, c1=1., c2=0.66)
        annulus_shape = shapes.annulus_vertices(0.15, 2.)
        agent_annulus = sprite.Sprite(
            x=0.5, y=0.5, shape=annulus_shape, scale=1., c0=0.6, c1=1., c2=1.)
        prey = get_prey(
            get_parallelogram(min_axis_ratio=0.5),
            scale=0.4,
            max_vel=max_vel,
            sprite_scale=0.075,
        )
        state = collections.OrderedDict([
            ('grid', grid),
            ('prey', prey),
            ('agent', [agent]),
            ('agent_annulus', [agent_annulus]),
        ])
        return state

    ############################################################################
    # Physics
    ############################################################################

    force = (physics_lib.Drag(coeff_friction=0.25), ['agent', 'agent_annulus'])
    corrective_physics = physics_lib.Tether(('prey',), update_angle_vel=True)
    physics = physics_lib.Physics(
        force,
        updates_per_env_step=10,
        corrective_physics=corrective_physics,
    )

    ############################################################################
    # Task
    ############################################################################

    prey_task = tasks.ContactReward(
        1, layers_0='agent', layers_1='prey',
        condition=lambda s_agent, s_prey: s_prey.c1 > 0.5,
    )
    reset_trial_task = tasks.Reset(
        condition=lambda state: all([s.c1 < 0.5 for s in state['prey']]),
        steps_after_condition=10,
    )
    task = tasks.CompositeTask(prey_task, reset_trial_task, timeout_steps=500)

    ############################################################################
    # Action space
    ############################################################################

    action_space = action_spaces.Joystick(
        scaling_factor=0.01, action_layers=('agent', 'agent_annulus'))

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

    # Make prey gray upon contact
    def _make_prey_gray(prey):
        prey.c1 = 0.
        prey.c2 = 0.6
    make_prey_gray = game_rules.ModifyOnContact(
        layers_0='agent',
        layers_1='prey',
        modifier_1=_make_prey_gray,
    )

    # Keep agent near center
    keep_near_center = game_rules.KeepNearCenter(
        agent_layer='agent',
        layers_to_center=['agent_annulus', 'prey'],
        grid_x=_GRID_SIZE,
    )

    rules = (make_prey_gray, keep_near_center)

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
        level: Int. Different values yield different velocities of the prey.
    """
    if level == 0:
        return _get_config(max_vel=0.)
    elif level == 1:
        return _get_config(max_vel=0.01)
    elif level == 2:
        return _get_config(max_vel=0.02)
    else:
        raise ValueError('Invalid level {}'.format(level))
