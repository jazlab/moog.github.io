"""Multi-object tracking with features.

In this task, the subject controls the position of a small cross. Typically,
this would be controlled by an eye-tracker to follow the subject's gaze. There
are a number of circles with oriented bars (each initially randomly vertical or
horizontal). Those circles with bars drift around an arena, bouncing off the
walls. At a random time, one of the bars rotates 90 degrees. The subject's goal
is to identify and fixate on that target.

After a brief initial period in which they are all visible, an occluder
appears occluding peripheral vision. This forces the subject to mentally keep
track of the targets' locations and orientations.
"""

import collections
import numpy as np

from moog import action_spaces
from moog import game_rules as gr
from moog import observers
from moog import physics as physics_lib
from moog import shapes
from moog import sprite
from moog import tasks
from moog.state_initialization import distributions as distribs

_FIXATION_THRESHOLD = 0.1


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


class ChangeTargetFeature(gr.AbstractRule):
    """Game rule to change the first target's oriented bar."""

    def step(self, state, meta_state):
        del meta_state
        """Rotate first target's bar np 90 degrees."""
        s = state['bars'][0]
        s.angle = s.angle + 0.5 * np.pi


def get_config(num_targets):
    """Get environment config.
    
    Args:
        num_targets: Int. Number of targets.
    """

    ############################################################################
    # Sprite initialization
    ############################################################################

    # Target circles
    target_factors = distribs.Product(
        [distribs.Continuous('x', 0.1, 0.9),
         distribs.Continuous('y', 0.1, 0.9),
         RadialVelocity(speed=0.01),],
        scale=0.1, shape='circle', c0=0., c1=0., c2=0.9,
    )

    # Target bars
    bar_factors = dict(
        scale=0.1, shape='square', aspect_ratio=0.3, c0=0., c1=0., c2=0.2)

    # Walls
    bottom_wall = [[-1, 0], [2, 0], [2, -1], [-1, -1]]
    top_wall = [[-1, 1], [2, 1], [2, 2], [-1, 2]]
    left_wall = [[0, -1], [0, 4], [-1, 4], [-1, -1]]
    right_wall = [[1, -1], [1, 4], [2, 4], [2, -1]]
    walls = [
        sprite.Sprite(shape=np.array(v), x=0, y=0, c0=0., c1=0., c2=0.5)
        for v in [bottom_wall, top_wall, left_wall, right_wall]
    ]

    # Occluder
    occluder_factors = dict(x=0.5, y=0.5, c0=0.6, c1=0.25, c2=0.5, opacity=0)

    # Cross shape for agent and fixation cross
    cross_shape = 0.1 * np.array([
        [-5, 1], [-1, 1], [-1, 5], [1, 5], [1, 1], [5, 1], [5, -1], [1, -1],
        [1, -5], [-1, -5], [-1, -1], [-5, -1]
    ])
    
    def state_initializer():

        fixation = sprite.Sprite(
            x=0.5, y=0.5, shape=cross_shape, scale=0.1, c0=0., c1=0., c2=0.)
        screen = sprite.Sprite(
            x=0.5, y=0.5, shape='square', scale=2., c0=0., c1=0., c2=1.)
            
        agent = sprite.Sprite(
            x=0.5, y=0.5, scale=0.04, shape=cross_shape, c0=0.33, c1=1., c2=1.)
        occluder_shape = shapes.annulus_vertices(0.13, 2.)
        occluder = sprite.Sprite(shape=occluder_shape, **occluder_factors)

        targets = [
            sprite.Sprite(**target_factors.sample())
            for _ in range(num_targets)
        ]

        bar_angles = 0.5 * np.pi * np.random.binomial(1, 0.5, (num_targets))
        bars = [
            sprite.Sprite(
                x=s.x, y=s.y, x_vel=s.x_vel, y_vel=s.y_vel, angle=angle,
                **bar_factors)
            for s, angle in zip(targets, bar_angles)
        ]

        state = collections.OrderedDict([
            ('walls', walls),
            ('targets', targets),
            ('bars', bars),
            ('occluder', [occluder]),
            ('screen', [screen]),
            ('fixation', [fixation]),
            ('agent', [agent]),
        ])
        return state

    ############################################################################
    # Physics
    ############################################################################

    elastic_collision = physics_lib.Collision(
        elasticity=1., symmetric=False, update_angle_vel=False)
    tether = physics_lib.TetherZippedLayers(
        layer_names=('targets', 'bars'), update_angle_vel=False)
    physics = physics_lib.Physics(
        (elastic_collision, 'targets', 'walls'),
        updates_per_env_step=10,
        corrective_physics=[tether],
    )

    ############################################################################
    # Task
    ############################################################################

    def _reward_condition(_, meta_state):
        return meta_state['phase'] == 'reward'
    task = tasks.Reset(
        condition=_reward_condition, reward_fn=lambda _: 1,
        steps_after_condition=10,
    )

    ############################################################################
    # Action space
    ############################################################################

    action_space = action_spaces.SetPosition(
        action_layers=('agent', 'occluder'))

    ############################################################################
    # Observer
    ############################################################################

    observer = observers.PILRenderer(
        image_size=(64, 64),
        anti_aliasing=1,
        color_to_rgb=observers.color_maps.hsv_to_rgb,
    )

    ############################################################################
    # Game rules
    ############################################################################

    # Fixation phase

    fixation_rule = gr.Fixation(
        'agent', 'fixation', _FIXATION_THRESHOLD, 'fixation_duration')
    def _should_end_fixation(_, meta_state):
        return meta_state['fixation_duration'] >= 15

    fixation_phase = gr.Phase(
        continual_rules=fixation_rule,
        end_condition=_should_end_fixation,
        name='fixation',
    )

    # Visible Phase
    
    vanish_fixation = gr.VanishByFilter('fixation', lambda _: True)
    vanish_screen = gr.VanishByFilter('screen', lambda _: True)

    visible_phase = gr.Phase(
        one_time_rules=[vanish_fixation, vanish_screen],
        duration=5,
        name='visible',
    )

    # Tracking Phase

    def _make_opaque(s):
        s.opacity = 255
    appear_occluder = gr.ModifySprites('occluder', _make_opaque)

    tracking_phase = gr.Phase(
        one_time_rules=appear_occluder,
        duration=lambda: np.random.randint(40, 80),
        name='tracking',
    )

    # Change Phase
    
    fixation_response_rule = gr.Fixation(
        'agent', 'targets', _FIXATION_THRESHOLD, 'response_duration')
    def _should_end_change(_, meta_state):
        return meta_state['response_duration'] >= 30

    change_phase = gr.Phase(
        one_time_rules=ChangeTargetFeature(),
        continual_rules=fixation_response_rule,
        name='change',
        end_condition=_should_end_change,
    )

    # Reward Phase

    def _make_transparent(s):
        s.opacity = 0
    disappear_occluder = gr.ModifySprites('occluder', _make_transparent)
    def _glue(s):
        s.velocity = np.zeros(2)
    glue_targets = gr.ModifySprites(('targets', 'bars'), _glue)
    
    reward_phase = gr.Phase(
        one_time_rules=(disappear_occluder, glue_targets),
        name='reward',
    )

    phase_sequence = gr.PhaseSequence(
        fixation_phase, visible_phase, tracking_phase, change_phase,
        reward_phase, meta_state_phase_name_key='phase',
    )

    ############################################################################
    # Final config
    ############################################################################

    config = {
        'state_initializer': state_initializer,
        'physics': physics,
        'task': task,
        'action_space': action_space,
        'observers': {'image': observer, 'state': observers.RawState()},
        'game_rules': (phase_sequence,),
        'meta_state_initializer': lambda: {'phase': ''},
    }
    return config
