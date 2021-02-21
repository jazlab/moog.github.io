"""Two-body distance-based forces."""

from . import abstract_force
import abc
import numpy as np


class DistanceForce(abstract_force.AbstractNewtonianForce,
                    metaclass=abc.ABCMeta):
    """Distance force base class.
    
    This class can be used for any force that attracts or repels two sprites as
    a function of their distance alone.
    """

    def __init__(self, force_fn, symmetric=False):
        """Constructor

        Args:
            force_fn: Force function that takes a scalar distance and produces a
                scalar force magnitude, the magnitude of the force to be applied
                between sprites at the given distance apart.
            symmetric: Bool. Whether to apply the force to both sprites
                involved. If False, only applies force to the second sprite, the
                second argument to .step().
        """
        self._force_fn = force_fn
        self._symmetric = symmetric

    def _compute_forces(self, sprite_0, sprite_1):
        """Compute forces on sprite_0 and sprite_1."""
        diff = sprite_1.position - sprite_0.position
        dist = np.linalg.norm(diff)
        if dist == 0.:
            return np.zeros(2), np.zeros(2)
        force_direction = diff / dist
        force_magnitude = self._force_fn(dist)
        sprite_1_force = force_magnitude * force_direction

        if self._symmetric:
            sprite_0_force = -1 * sprite_1_force
        else:
            sprite_0_force = np.array([0., 0.])

        return sprite_0_force, sprite_1_force


def linear_force_fn(zero_intercept,
                    slope,
                    apply_distant_force=False,
                    apply_nearby_force=True):
    """Force is a linear function of distance.
    
    Args:
        zero_intercept: Scalar. Magnitude of force between sprites a distance
            zero apart.
        slope: Scalar. Slope of linear change in force with respect to distance.
        apply_distant_force: Bool. Whether to apply force at distances greater
            than the event horizon (distance at which the force is zero).
        apply_distant_force: Bool. Whether to apply force at distances less than
            the event horizon.

    Returns:
        force_fn: Function distance -> force magnitude.
    """
    event_horizon = -1. * zero_intercept / slope
    def force_fn(distance):
        force_magnitude = zero_intercept + slope * distance
        if not apply_distant_force and distance > event_horizon:
            force_magnitude = 0
        if not apply_nearby_force and distance < event_horizon:
            force_magnitude = 0
        return force_magnitude
    return force_fn


def spring_force_fn(spring_constant, equilibrium=0):
    """Spring force by Hooke's Law.
    
    Args:
        spring_constant: Scalar. Spring constant.
        equilibrium: Spring equilibrium. Distance at which the force is zero.

    Returns:
        force_fn: Function distance -> force magnitude.
    """
    def force_fn(distance):
        return -1. * spring_constant * (distance - equilibrium)
    return force_fn
