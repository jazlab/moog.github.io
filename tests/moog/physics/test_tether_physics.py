"""Tests for moog/physics/tether_physics.py.

To run this test, navigate to this directory and run
```bash
$ pytest test_tether_physics.py --capture=tee-sys
```

Note: The --capture=tee-sys routes print statements to stdout, which is useful
for debugging.

Alternatively, to run this test and any others, navigate to any parent directory
and simply run
```bash
$ pytest --capture=tee-sys
```
This will run all test_* files in children directories.
"""

import sys
sys.path.insert(0, '...')  # Allow imports from moog codebase

import collections
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pytest

from moog import physics as physics_lib
from moog import shapes
from moog import sprite
from moog.physics import collisions
from moog.observers import pil_renderer

# Absolute error tolerance for testing scalars (positions, velocities, etc.)
_ATOL = 0.001


class MatplotlibUI():
    """Matplotlib UI.
    
    This can be used to visualize test conditions.
    """

    def __init__(self, image_size=128, anti_aliasing=3):
        """Constructor."""
        plt.ion()
        self._fig, self._ax = plt.subplots()
        self._ax.axis('off')
        self._imshow = self._ax.imshow(
            np.zeros((image_size, image_size, 3)), interpolation='none')
        self._renderer = pil_renderer.PILRenderer(
            image_size=(image_size, image_size), anti_aliasing=anti_aliasing)
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        plt.show(block=False)
        plt.pause(0.1)

    def _render(self, state):
        """Renderer a state (ordereddict of iterables of sprites)."""
        self._imshow.set_data(self._renderer(state))
        plt.draw()
        plt.pause(0.1)

    def _simulate_video(self, physics, state, steps):
        """Simulate and display video."""
        physics.step(state)
        for i, s in enumerate(state['sprites']):
            print('')
            print(f'Sprite {i}')
            print(f'position: [{s.position[0]:.4f}, {s.position[1]:.4f}]')
            print(f'velocity: [{s.velocity[0]:.4f}, {s.velocity[1]:.4f}]')
            print(f'angle_vel: {s.angle_vel:.4f}')

        for _ in range(steps - 1):
            physics.step(state)
            self._render(state)
        
        # Print the true positions, velocities, and angular velocities of the
        # sprites, rounded to 4 decimal places.
        for i, s in enumerate(state['sprites']):
            print('')
            print(f'Sprite {i}')
            print(f'position: [{s.position[0]:.4f}, {s.position[1]:.4f}]')
            print(f'velocity: [{s.velocity[0]:.4f}, {s.velocity[1]:.4f}]')
            print(f'angle_vel: {s.angle_vel:.4f}')


def get_state():
    """Get initial state."""
    sprite_0 = sprite.Sprite(
        x=0.5, y=0.7, scale=0.1, shape='triangle',
        x_vel=0.04, y_vel=-0.02, c0=255, angle=2.)
    sprite_1 = sprite.Sprite(
        x=0.2, y=0.6, scale=0.1, shape='triangle',
        x_vel=0., y_vel=0., c1=255, angle=1.)
    sprite_2 = sprite.Sprite(
        x=0.6, y=0.3, scale=0.1, shape='triangle',
        x_vel=0., y_vel=0., c2=255)
    sprites = [sprite_0, sprite_1, sprite_2]

    walls = shapes.border_walls(
        visible_thickness=0.05, c0=128, c1=128, c2=128)

    state = collections.OrderedDict([('walls', walls), ('sprites', sprites)])
    
    return state


class TestTether():
    """Test Tether."""

    def _run_test(self, tether, step_1_state, final_state, plot=False):
        """Run test.
        
        Set plot = True to display videos of the test conditions.
        
        Args:
            tether: Tether force.
            step_1_state: Iterable of lists, one for each sprite. Each element
                is a list of [position, velocity, angle_vel] for the sprite
                after the first physics step.
            final_state: Same as step_1_state, except for the final step.
            plot: Bool. Whether to display video or run test.
        """
        state = get_state()

        collision = physics_lib.Collision(
            elasticity=0., symmetric=False, update_angle_vel=True)
        physics = physics_lib.Physics(
            (collision, 'sprites', 'walls'),
            corrective_physics=[tether],
            updates_per_env_step=10,
        )

        steps = 45
        if plot:
            MatplotlibUI()._simulate_video(physics, state, steps=steps)
        else:
            physics.step(state)

            for s, pred in zip(state['sprites'], step_1_state):
                assert np.allclose(s.position, pred[0], atol=_ATOL)
                assert np.allclose(s.velocity, pred[1], atol=_ATOL)
                assert np.allclose(s.angle_vel, pred[2], atol=_ATOL)

            for _ in range(steps - 1):
                physics.step(state)
            
            for s, pred in zip(state['sprites'], final_state):
                assert np.allclose(s.position, pred[0], atol=_ATOL)
                assert np.allclose(s.velocity, pred[1], atol=_ATOL)
                assert np.allclose(s.angle_vel, pred[2], atol=_ATOL)

    def testTetherNoAngleUpdate(self, plot=False):
        """Tether.
        
        Set plot = True to display videos of the test conditions.
        """
        tether = physics_lib.Tether('sprites', update_angle_vel=False)

        step_1_state = [
            [[0.5133, 0.6933], [0.0133, -0.0067], 0.],
            [[0.2133, 0.5933], [0.0133, -0.0067], 0.],
            [[0.6133, 0.2933], [0.0133, -0.0067], 0.],
        ]

        final_state = [
            [[0.7710, 0.4900], [-0.0005, 0.], 0.],
            [[0.4710, 0.3900], [-0.0005, 0.], 0.],
            [[0.8671, 0.0939], [-0.0005, 0.], 0.],
        ]

        self._run_test(tether, step_1_state, final_state, plot=plot)

    def testTetherAngleUpdate(self, plot=False):
        """Tether.
        
        Set plot = True to display videos of the test conditions.
        """
        tether = physics_lib.Tether('sprites', update_angle_vel=True)

        step_1_state = [
            [[0.5206, 0.6900], [0.0205, -0.0103], -0.0447],
            [[0.2165, 0.6035], [0.0166, 0.0033], -0.0447],
            [[0.6027, 0.2860], [0.0025, -0.0140], -0.0447],
        ]

        final_state = [
            [[0.8341, 0.3401], [-0.0028, -0.0046], -0.0229],
            [[0.7139, 0.6271], [0.0037, -0.0018], -0.0229],
            [[0.4545, 0.2062], [-0.0059, 0.0041], -0.0229],
        ]

        self._run_test(tether, step_1_state, final_state, plot=plot)

    def testTetherAnchor(self, plot=False):
        """Tether.
        
        Set plot = True to display videos of the test conditions.
        """
        tether = physics_lib.Tether('sprites', anchor=np.array([0.2, 0.2]))

        step_1_state = [
            [[0.5190, 0.6881], [0.0188, -0.0122], -0.0385],
            [[0.2154, 0.5997], [0.0154, -0.0006], -0.0385],
            [[0.6036, 0.2845], [0.0033, -0.0155], -0.0385],
        ]

        final_state = [
            [[0.6927, 0.5118], [0., 0.], 0.],
            [[0.3798, 0.5573], [0., 0.], 0.],
            [[0.6025, 0.1233], [0., 0.], 0.]
        ]

        self._run_test(tether, step_1_state, final_state, plot=plot)

