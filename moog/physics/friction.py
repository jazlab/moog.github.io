"""Friction forces.

This file has two classes, KineticFriction and Drag. The only difference between
the two is that kinetic friction is velocity-independent, whereas drag scales
linearly with velocity. For joystick control of agents, using Drag is a bit
easier since it prevents the velocity from exploding.
"""

from . import abstract_force
import numpy as np


class KineticFriction(abstract_force.AbstractNewtonianForce):
    """Kinetic friction force class."""

    def __init__(self, coeff_friction=1.):
        """Constructor.

        Args:
            coeff_friction: Scalar. Coefficient of friction. The frictional
                force applied to a sprite is -coeff_friction * sprite.mass.
        """
        self._coeff_friction = coeff_friction

    def _compute_forces(self, sprite):
        velocity_norm = np.linalg.norm(sprite.velocity)
        if velocity_norm == 0:
            normalized_velocity = np.zeros(2)
        else:
            normalized_velocity = sprite.velocity / velocity_norm
        force = -1 * self._coeff_friction * normalized_velocity * sprite.mass

        return (force,)
        

class Drag(abstract_force.AbstractNewtonianForce):
    """Drag force class.
    
    Note: The drag force implemented here depends only on {velocity, mass,
    coef_friction}, not on cross sectional area, so it is not a realistic drag
    force.
    """

    def __init__(self, coeff_friction=1.):
        """Constructor.

        Args:
            coeff_friction: Scalar. Coefficient of friction. The frictional
                force applied to a sprite is
                -coeff_friction * sprite.velocity * sprite.mass.
        """
        self._coeff_friction = coeff_friction

    def _compute_forces(self, sprite):
        force = -1 * self._coeff_friction * sprite.velocity * sprite.mass
        return (force,)
