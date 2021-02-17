"""Gravity forces."""

from . import abstract_force
import numpy as np


class DownGravity(abstract_force.AbstractNewtonianForce):
    """Force class for downwards gravity.
    
    This gravity acts to pull objects down, as if on Earth's surface.
    """

    def __init__(self, g=-1.):
        """Constructor.

        Args:
            g: Scalar. Gravitational constant.
        """
        self._g = g

    def _compute_forces(self, sprite):
        force = self._g * sprite.mass * np.array([0, 1])
        return (force,)


class Gravity(abstract_force.AbstractNewtonianForce):
    """Force class for gravity.
    
    This gravity acts between bodies in space.
    """

    def __init__(self, g=-1., symmetric=True):
        """Constructor.

        Args:
            g: Scalar. Gravitational constant.
            symmetric: Bool. Whether to apply the force to both sprites
                involved. If False, only applies force to the second sprite, the
                second argument to .step().
        """
        self._g = g
        self._symmetric = symmetric

    def _compute_forces(self, sprite_0, sprite_1):
        """Compute forces on sprite_0 and sprite_1."""
        diff = sprite_1.position - sprite_0.position
        dist = np.linalg.norm(diff)
        if dist == 0.:
            return np.zeros(2), np.zeros(2)
        force_direction = diff / dist
        force_magnitude = self._g * sprite_0.mass * sprite_1.mass * dist

        sprite_1_force = force_magnitude * force_direction

        if self._symmetric:
            sprite_0_force = -1 * sprite_1_force
        else:
            sprite_0_force = np.array([0., 0.])

        return sprite_0_force, sprite_1_force
