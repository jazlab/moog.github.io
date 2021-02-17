"""Tests for moog/physics/collisions.py.

To run this test, navigate to this directory and run
```bash
$ pytest test_collisions.py --capture=tee-sys
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

from moog import sprite
from moog.physics import collisions
from moog.observers import pil_renderer

# Absolute error tolerance for testing scalars (positions, velocities, etc.)
_ATOL = 0.001


def _apply_pairwise_force(sprites, force, symmetric):
    """Apply pair-wise force to sprites, updating the sprites in-place.

    Args:
        sprites: Iterable of instances of sprite.Sprite.
        force: Instance of a force.
        symmetric: Bool. Whether to apply force symmetrically. Only really makes
            sense when len(sprites) == 2.
    """
    num_sprites = len(sprites)
    if symmetric:
        inds = [(i, j) for j in range(num_sprites) for i in range(num_sprites)]
    else:
        inds = [(i, j) for i in range(num_sprites) for j in range(i)]
    for i, j in inds:
        force.step(sprites[i], sprites[j], updates_per_env_step=1)


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

    def _simulate_video(self, sprites, force, steps, symmetric):
        """Simulate and display video."""
        for _ in range(steps):
            self._render(collections.OrderedDict([('', sprites)]))
            _apply_pairwise_force(sprites, force, symmetric=symmetric)
            for s in sprites:
                s.update_pos_from_vel(delta_t=1.)
        
        # Print the true positions, velocities, and angular velocities of the
        # sprites, rounded to 4 decimal places.
        for i, s in enumerate(sprites):
            print('')
            print(f'Sprite {i}')
            print(f'position: [{s.position[0]:.4f}, {s.position[1]:.4f}]')
            print(f'velocity: [{s.velocity[0]:.4f}, {s.velocity[1]:.4f}]')
            print(f'angle_vel: {s.angle_vel:.4f}')


class TestCollisions():
    """Test collisions."""

    @pytest.mark.parametrize(
        ('init_pos_0, init_vel_0, out_pos_0, out_vel_0, out_pos_1, out_vel_1, '
         'elasticity, symmetric'),
        [
            # Head-on, asymmetric
            ([0.5, 0.35], [0., 0.], [0.5, 0.35], [0., 0.], [0.5, 0.4827],
             [0., 0.01], 1., False),
            # Head-on, symmetric
            ([0.5, 0.35], [0., 0.], [0.5, 0.3287], [0., -0.01], [0.5, 0.4613],
             [0., 0.], 1., True),
            # Head-on, symmetric, semi-elastic
            ([0.5, 0.35], [0., 0.], [0.5, 0.3337], [0., -0.0075], [0.5, 0.4563],
             [0., -0.0025], 0.5, True),
            # Head-on, symmetric, inelastic
            ([0.5, 0.35], [0., 0.], [0.5, 0.3387], [0., -0.005], [0.5, 0.4513],
             [0., -0.005], 0., True),
            # Head-on, symmetric, both moving
            ([0.5, 0.35], [0., 0.01], [0.5, 0.3287], [0., -0.01], [0.5, 0.5213],
             [0., 0.01], 1., True),
            # Offset, asymmetric
            ([0.44, 0.37], [0., 0.], [0.44, 0.37], [0., 0.], [0.5217, 0.4699],
             [0.0095, 0.0031], 1., False),
            # Offset, symmetric
            ([0.44, 0.37], [0., 0.], [0.4291, 0.3550], [-0.0048, -0.0065],
             [0.5109, 0.4550], [0.0048, -0.0035], 1., True),
            # Offset, symmetric, semi-elastic
            ([0.44, 0.37], [0., 0.], [0.4315, 0.3583], [-0.0036, -0.0049],
             [0.5085, 0.4517], [0.0036, -0.0051], 0.5, True),
            # Offset, symmetric, both moving
            ([0.44, 0.37], [0., 0.01], [0.4006, 0.3758], [-0.0095, -0.0031],
             [0.5394, 0.4942], [0.0095, 0.0031], 1., True),
            # Offset, symmetric, both moving, diagonal motion
            ([0.43, 0.36], [0.015, 0.01], [0.4793, 0.3286], [0.0051, -0.0123],
             [0.5407, 0.5314], [0.0099, 0.0123], 1., True),
        ]
    )
    def testCirclesSameMass(self,
                            init_pos_0,
                            init_vel_0,
                            out_pos_0,
                            out_vel_0,
                            out_pos_1,
                            out_vel_1,
                            elasticity,
                            symmetric,
                            steps=6,
                            plot=False):
        """Two circles with same size and mass colliding.
        
        Set plot = True to display videos of the test conditions.
        """
        force = collisions.Collision(
            elasticity=elasticity, symmetric=symmetric, update_angle_vel=False)
        sprite_0 = sprite.Sprite(
            x=init_pos_0[0], y=init_pos_0[1], scale=0.1, shape='circle',
            x_vel=init_vel_0[0], y_vel=init_vel_0[1], c1=255)
        sprite_1 = sprite.Sprite(
            x=0.5, y=0.5, scale=0.1, shape='circle', y_vel=-0.01, c0=255)
        
        if plot:
            MatplotlibUI()._simulate_video(
                [sprite_0, sprite_1], force, steps, symmetric)
        else:
            for _ in range(steps):
                _apply_pairwise_force(
                    [sprite_0, sprite_1], force, symmetric=symmetric)
                sprite_0.update_pos_from_vel(delta_t=1.)
                sprite_1.update_pos_from_vel(delta_t=1.)
            
            assert np.allclose(sprite_0.position, out_pos_0, atol=_ATOL)
            assert np.allclose(sprite_0.velocity, out_vel_0, atol=_ATOL)
            assert np.allclose(sprite_1.position, out_pos_1, atol=_ATOL)
            assert np.allclose(sprite_1.velocity, out_vel_1, atol=_ATOL)

    @pytest.mark.parametrize(
        'init_pos_0, init_vel_0, out_pos_0, out_vel_0, out_pos_1, out_vel_1',
        [
            # Head-on
            ([0.5, 0.35], [0., 0.], [0.5, 0.3220], [0., -0.0133], [0.5, 0.4547],
             [0., -0.0033]),
            # Head-on, semi-elastic
            ([0.5, 0.35], [0., 0.], [0.5, 0.3220], [0., -0.0133],
             [0.5, 0.4547], [0., -0.0033]),
            # Offset, symmetric, both moving
            ([0.44, 0.37], [0., 0.01], [0.3879, 0.3583], [-0.0127, -0.0075],
             [0.5267, 0.4768], [0.0063, -0.0013]),
            # Offset, symmetric, both moving, diagonal motion
            ([0.43, 0.36], [0.015, 0.01], [0.4661, 0.2989], [0.0018, -0.0197],
             [0.5275, 0.5017], [0.0066, 0.0048]),
        ]
    )
    def testCirclesDifferentMass(self,
                                 init_pos_0,
                                 init_vel_0,
                                 out_pos_0,
                                 out_vel_0,
                                 out_pos_1,
                                 out_vel_1,
                                 steps=6,
                                 plot=False):
        """Two circles with same size and different massed colliding.
        
        Set plot = True to display videos of the test conditions.
        """
        force = collisions.Collision(
            elasticity=1., symmetric=True, update_angle_vel=False)
        sprite_0 = sprite.Sprite(
            x=init_pos_0[0], y=init_pos_0[1], scale=0.1, shape='circle',
            x_vel=init_vel_0[0], y_vel=init_vel_0[1], c1=255)
        sprite_1 = sprite.Sprite(
            x=0.5, y=0.5, scale=0.1, shape='circle', y_vel=-0.01, c0=255,
            mass=2.)
        
        if plot:
            MatplotlibUI()._simulate_video(
                [sprite_0, sprite_1], force, steps, symmetric=True)
        else:
            for _ in range(steps):
                _apply_pairwise_force(
                    [sprite_0, sprite_1], force, symmetric=True)
                sprite_0.update_pos_from_vel(delta_t=1.)
                sprite_1.update_pos_from_vel(delta_t=1.)
            
            assert np.allclose(sprite_0.position, out_pos_0, atol=_ATOL)
            assert np.allclose(sprite_0.velocity, out_vel_0, atol=_ATOL)
            assert np.allclose(sprite_1.position, out_pos_1, atol=_ATOL)
            assert np.allclose(sprite_1.velocity, out_vel_1, atol=_ATOL)

    @pytest.mark.parametrize(
        ('init_angle_vel_0, out_pos_0, out_vel_0, out_angle_vel_0, out_pos_1, '
         'out_vel_1, out_angle_vel_1, elasticity, update_angle_vel'),
        [
            # No angular velocity, elastic, no update_angle_vel
            (0., [0.5064, 0.6776], [-0.0044, 0.0024], 0., [0.6369, 0.5358],
             [0.0044, -0.0024], 0., 1., False),
            # No angular velocity, elastic
            (0., [0.5411, 0.6689], [0.0025, 0.0006], -0.0911, [0.6022, 0.5444],
             [-0.0025, -0.0006], 0.0362, 1., True),
            # No angular velocity, semi-elastic
            (0., [0.5442, 0.6681], [0.0031, 0.0005], -0.0683, [0.5991, 0.5452],
             [-0.0031, -0.0005], 0.0271, 0.5, True),
            # Positive angular velocity, elastic
            (0.1, [0.4950, 0.6804], [-0.0021, 0.0018], -0.1215, [0.6483, 0.5329],
             [0.0021, -0.0018], 0.0720, 1., True),
            # Negative angular velocity, elastic
            (-0.02, [0.5486, 0.6670], [0.0035, 0.0004], -0.0800, [0.5947, 0.5463],
             [-0.0035, -0.0004], 0.0250, 1., True),
        ]
    )
    def testTriangles(self,
                      init_angle_vel_0,
                      out_pos_0,
                      out_vel_0,
                      out_angle_vel_0,
                      out_pos_1,
                      out_vel_1,
                      out_angle_vel_1,
                      elasticity,
                      update_angle_vel,
                      steps=10,
                      plot=False):
        """Two irregular triangles.
        
        Set plot = True to display videos of the test conditions.
        """
        force = collisions.Collision(
            elasticity=elasticity, symmetric=True,
            update_angle_vel=update_angle_vel)
        shape_0 = np.array([[1, 1], [1, 3], [-2, -2]])
        sprite_0 = sprite.Sprite(
            x=0.5, y=0, scale=0.05, shape=shape_0, x_vel=0.005, y_vel=0.,
            c0=255, angle=1., angle_vel=init_angle_vel_0)
        shape_1 = np.array([[2, 1], [0, 1], [-1, -3]])
        sprite_1 = sprite.Sprite(
            x=0.31, y=0.88, scale=0.05, shape=shape_1, x_vel=-0.005, y_vel=0.,
            c1=255)
        
        if plot:
            MatplotlibUI()._simulate_video(
                [sprite_0, sprite_1], force, steps, symmetric=True)
        else:
            for _ in range(steps):
                _apply_pairwise_force(
                    [sprite_0, sprite_1], force, symmetric=True)
                sprite_0.update_pos_from_vel(delta_t=1.)
                sprite_1.update_pos_from_vel(delta_t=1.)
            
            assert np.allclose(sprite_0.position, out_pos_0, atol=_ATOL)
            assert np.allclose(sprite_0.velocity, out_vel_0, atol=_ATOL)
            assert np.allclose(sprite_0.angle_vel, out_angle_vel_0, atol=_ATOL)
            assert np.allclose(sprite_1.position, out_pos_1, atol=_ATOL)
            assert np.allclose(sprite_1.velocity, out_vel_1, atol=_ATOL)
            assert np.allclose(sprite_1.angle_vel, out_angle_vel_1, atol=_ATOL)
