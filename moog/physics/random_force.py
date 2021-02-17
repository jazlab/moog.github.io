"""Random force."""

from . import abstract_force
import numpy as np


class RandomForce(abstract_force.AbstractNewtonianForce):
    """Random force class."""

    def __init__(self, max_force_magnitude):
        """Constructor.

        This class produces a force with magnitude uniformly sampled in
        [0, max_force_magnitude] and angle of direction randomly uniformly
        sampled in [0, 2 * pi].

        Args:
            max_force_magnitude: Scalar. Maximum force magnitude.
        """
        self._max_force_magnitude = max_force_magnitude

    def _compute_forces(self, sprite):
        r = np.random.uniform(0, self._max_force_magnitude)
        theta = np.random.uniform(0, 2 * np.pi)
        force = np.array([r * np.cos(theta), r * np.sin(theta)])
        return (force,)
