"""Constant speed corrective physics.

The ConstantSpeed class is a physics object that forces sprites in specified
layers to have constant speed. It is typically used as a corrective physics.
"""

from . import abstract_physics
import itertools
import numpy as np


class ConstantSpeed(abstract_physics.AbstractPhysics):
    """Constant speed physics class.
    
    Forces sprites in specified layers to have constant speed by normalizing
    their velocities.
    """

    def __init__(self, layer_names, speed):
        """Constructor.

        Args:
            layer_names: String or iterable of strings corresponding to layers
                in environment state. All sprites in these layers will have
                speed equal to `speed` argument.
            speed: Float. Sprite velocities will be normalized to have norm
                equal to this (or zero if their velocity is zero).
        """
        if not isinstance(layer_names, (list, tuple)):
            layer_names = [layer_names]
        self._layer_names = layer_names
        self._speed = speed

    def apply_physics(self, state, updates_per_env_step):
        """Apply the physics.
        
        Args:
            state: Environment state.
            updates_per_env_step: Int. Number of times the physics is applied
                per environment step.
        """
        sprites = [s for k in self._layer_names for s in state[k]]
        for s in sprites:
            norm_velocity = np.linalg.norm(s.velocity)
            if norm_velocity:
                s.velocity = self._speed * s.velocity / norm_velocity
