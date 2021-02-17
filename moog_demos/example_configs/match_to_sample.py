"""Match-to-sample task.

In this task there are some colored circles (targets) on a ring, and an agent
avatar in the center of the ring. After an initial stimulus period where the
colored targets are visible, they all turn grey and begin rotating together.
After some time they stop and a colored cue appears on the agent avatar. The
subject must identify the target that had the same color as the cue and respond
by navigating towards that target.

This task requires workimg memory of multiple objects with features (colors).
"""

import collections
import copy
import functools
import numpy as np

from moog import action_spaces
from moog import game_rules as gr
from moog import observers
from moog import physics as physics_lib
from moog import shapes
from moog import sprite
from moog import tasks
from moog.state_initialization import distributions as distribs


def _get_polygon(num_sides, min_angle):
    """Get polygon vertices centered around 0 with radius 1."""
    # Get vertex angles
    angles = [0.]
    def _get_random_angle():
        return np.random.uniform(min_angle, 2 * np.pi - min_angle)
    for _ in range(num_sides - 1):
        valid_angle = False
        while not valid_angle:
            angle = _get_random_angle()
            if all([np.abs(angle - a) > min_angle for a in angles]):
                valid_angle = True
        angles.append(angle)

    # Sort and randomly rotate the angles
    angles = np.sort(angles)
    angles += np.random.uniform(0., 2 * np.pi)

    # Generate the vertices
    vertices = np.stack((np.sin(angles), np.cos(angles)), axis=1)

    return vertices


class BeginMotion(gr.AbstractRule):
    """Custom rule for motion initiation."""
    
    def __init__(self, angle_vel_range):
        """Constructor.

        Args:
            angle_vel_range: Tuple of non-negative floats. Range of angular
                velocity magnitudes, in radians per step.
        """
        self._angle_vel_range = angle_vel_range

    def step(self, state, meta_state):
        del meta_state
        targets = state['targets']
        covers = state['covers']
        angle_vel = np.random.uniform(*self._angle_vel_range)
        angle_vel *= (2 * np.random.randint(2) - 1)
        for t, c in zip(targets, covers):
            relative_pos = t.position - 0.5
            perpendicular = np.matmul(np.array([[0, -1], [1, 0]]), relative_pos)
            radius = np.linalg.norm(relative_pos)
            vel = perpendicular * radius * angle_vel
            t.velocity = vel
            c.velocity = vel


def get_config(num_targets):
    """Get environment config.
    
    Args:
        num_targets: Int. Number of targets.
    """
    if num_targets == 0 or not isinstance(num_targets, int):
        raise ValueError(
            f'num_targets is {num_targets}, but must be a positive integer')

    ############################################################################
    # State initialization
    ############################################################################

    screen = sprite.Sprite(
        x=0.5, y=0.5, shape='square', scale=2., c0=0.6, c1=0.7, c2=0.7)
        
    target_factor_distrib = distribs.Product(
        [distribs.Continuous('c0', 0., 1.)],
        shape='circle', scale=0.085, c1=1., c2=1.,
    )
    cover_factors = dict(
        mass=0., shape='circle', scale=0.1, c0=0., c1=0., c2=0.5, opacity=0)
    
    def state_initializer():
        """State initializer method to be fed into environment."""

        # Get targets and covers
        sprite_positions = 0.5 + 0.35 * _get_polygon(num_targets, 0.7)
        target_factors = [
            target_factor_distrib.sample() for _ in range(num_targets)]
        targets = [
            sprite.Sprite(x=pos[0], y=pos[1], **factors)
            for pos, factors in zip(sprite_positions, target_factors)
        ]
        covers = [
            sprite.Sprite(x=pos[0], y=pos[1], **cover_factors)
            for pos in sprite_positions
        ]

        # Tag the cover metadata based on whether they are prey or not
        for i, s in enumerate(covers):
            if i == 0:
                s.metadata = {'prey': True}
            else:
                s.metadata = {'prey': False}
        
        # Make cue have the same factors as the first target, except slightly
        # smaller
        cue_factors = copy.deepcopy(target_factors[0])
        cue_factors['scale'] = 0.7 * target_factors[0]['scale']
        cue = sprite.Sprite(
            x=0.5, y=0.501, opacity=0, mass=np.inf, **cue_factors)

        agent = sprite.Sprite(
            x=0.5, y=0.5, shape='circle', scale=0.1, c0=0.4, c1=0., c2=1.,
            mass=np.inf)
        annulus_verts = shapes.annulus_vertices(0.34, 0.36)
        annulus = sprite.Sprite(
            x=0.5, y=0.5, shape=annulus_verts, scale=1., c0=0., c1=0., c2=0.3)
        
        state = collections.OrderedDict([
            ('annulus', [annulus]),
            ('targets', targets),
            ('covers', covers),
            ('agent', [agent]),
            ('cue', [cue]),
            ('screen', [screen]),
        ])
        return state

    ################################################################################
    # Physics
    ################################################################################

    drag = (physics_lib.Drag(coeff_friction=0.25), ['agent', 'cue'])
    tether_covers = physics_lib.TetherZippedLayers(
        ('targets', 'covers'), anchor=np.array([0.5, 0.5]))
    physics = physics_lib.Physics(
        drag,
        updates_per_env_step=1,
        corrective_physics=[tether_covers],
    )

    ################################################################################
    # Task
    ################################################################################

    contact_task = tasks.ContactReward(
        reward_fn=lambda _, s: 1 if s.metadata['prey'] else -1,
        layers_0='agent',
        layers_1='covers',
    )

    def _should_reset(state, meta_state):
        should_reset = (
            state['covers'][0].opacity == 0 and
            meta_state['phase'] == 'response'
        )
        return should_reset
    reset_task = tasks.Reset(
        condition=_should_reset,
        steps_after_condition=15,
    )

    task = tasks.CompositeTask(contact_task, reset_task, timeout_steps=800)
    
    ################################################################################
    # Action Space
    ################################################################################

    action_space = action_spaces.Joystick(
        scaling_factor=0.01, action_layers=['agent', 'cue'])

    ################################################################################
    # Observer
    ################################################################################

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

    def _make_opaque(s):
        s.opacity = 255
    def _make_transparent(s):
        s.opacity = 0

    # Screen Phase

    screen_phase = gr.Phase(duration=1, name='screen')

    # Visible Phase

    disappear_screen = gr.ModifySprites('screen', _make_transparent)
    visible_phase = gr.Phase(
        one_time_rules=disappear_screen, duration=2, name='visible')

    # Motion Phase

    def _move(s):
        s.velocity = np.random.uniform(-0.25, 0.25, size=(2,))

    cover_targets = gr.ModifySprites('covers', _make_opaque)
    begin_motion = BeginMotion(angle_vel_range=(0.1, 0.3))
    motion_phase = gr.Phase(
        one_time_rules=[cover_targets, begin_motion],
        duration=100,
        name='motion',
    )

    # Response Phase

    def _stop(s):
        s.angle_vel = 0.
        s.velocity = np.zeros(2)
    def _unglue(s):
        s.mass = 1.

    appear_cue = gr.ModifySprites('cue', _make_opaque)
    stop_targets = gr.ModifySprites(('targets', 'covers'), _stop)
    unglue_agent = gr.ModifySprites(('agent', 'cue'), _unglue)
    make_targets_discoverable = gr.ModifyOnContact(
        layers_0='agent', layers_1='covers', modifier_1=_make_transparent)

    response_phase = gr.Phase(
        one_time_rules=[appear_cue, stop_targets, unglue_agent],
        continual_rules=make_targets_discoverable,
        name='response',
    )

    phase_sequence = gr.PhaseSequence(
        screen_phase, visible_phase, motion_phase, response_phase,
        meta_state_phase_name_key='phase',
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
        'game_rules': (phase_sequence,),
        'meta_state_initializer': lambda: {'phase': ''}
    }
    return config
