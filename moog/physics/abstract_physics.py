"""Abstract physic class."""

import abc


class AbstractPhysics(abc.ABC):
    """Abstract physics class.
    
    All physics objects must inherit from this class.
    """

    def __init__(self, updates_per_env_step):
        self._updates_per_env_step = updates_per_env_step
    
    @abc.abstractmethod
    def apply_physics(self, state, updates_per_env_step):
        """Step the physics.
        
        Args:
            state: Environment state.
            updates_per_env_step: Int. Number of times the physics is applied
                per environment step.
        """
        pass

    def reset(self, state):
        """Reset the physics.

        This is called at every episode reset of the environment. Usually it
        does nothing (usually the physics don't change between episodes), but
        for maze environments this is used to initialize the maze for the
        episode.

        Args:
            state: Environment state.
        """
        pass

    def step(self, state):
        """Step physics, applying forces self._updates_per_env_step times."""
        for _ in range(self._updates_per_env_step):
            self.apply_physics(state, self._updates_per_env_step)

    @property
    def updates_per_env_step(self):
        """Number of times the physics is applied per environment step."""
        return self._updates_per_env_step

