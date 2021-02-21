"""Abstract force classes.

This file contains AbstractForce and AbstraceNewtonianForce classes.
"""

import abc
import numpy as np


class AbstractForce(abc.ABC):
    """Abstract force class.
    
    All physics forces must inherit from this class.
    """
    
    @abc.abstractmethod
    def step(self, *sprites, updates_per_env_step):
        """Step the force.
        
        Args:
            *sprites: Iterable of sprites on which the force operates.
            updates_per_env_step: Int. Number of times this force step is called
                for each step of the physics in the environment. This is used
                for example to scale down Netwonian forces so that from the
                user's perspective the force is invariant to
                updates_per_env_step.
        """
        pass

    def reset(self, state):
        """Reset the force.

        This is called at every episode reset of the environment. Usually it
        does nothing (usually the forces don't change between episodes), but
        for maze environments this is used to initialize the maze for the
        episode.

        Args:
            state: Environment state.
        """
        pass


class AbstractNewtonianForce(AbstractForce, metaclass=abc.ABCMeta):
    """Abstract Newtonian force class.
    
    Forces abiding by F = ma should inherit from this class.
    """

    @abc.abstractmethod
    def _compute_forces(self, *sprites):
        """Compute forces on all sprites.
        
        Args:
            *sprites: Iterable of sprites on which the force operates.
        
        Returns:
            Iterable with same length as sprites. Each element is a scalar numpy
            array [f_x, f_y], the force to be applied to the corresponding
            sprite.
        """
        pass
    
    def step(self, *sprites, updates_per_env_step):
        forces = self._compute_forces(*sprites)
        for sprite, force in zip(sprites, forces):
            if not np.isfinite(sprite.mass):
                # Good to catch this because sometimes we might make a sprite
                # have infinite mass to prevent it from moving, but we don't
                # want that sprite's velocity and consequently position to
                # become NaN.
                continue
            delta_vel = force / (sprite.mass * float(updates_per_env_step))
            sprite.velocity += delta_vel
