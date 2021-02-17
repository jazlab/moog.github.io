"""Abstract action space."""

import abc


class AbstractActionSpace(abc.ABC):
    """Abstract action space class.
    
    All action spaces must inherit from this class.
    """
    
    @abc.abstractmethod
    def step(self, state, action):
        """Apply action to environment state.

        The action may do things like change sprites' positions or velocities,
        etc. Any change to the state of the environment can be implemented by an
        action space.
        
        Args:
            state: Environment state. OrderedDict of iterables of sprites.
            action: Action object. Type and size depends on action space.
        """
        pass

    def reset(self, state):
        """Reset action space at start of new episode.

        Action spaces are often state-less, but may be stateful (e.g. have a
        momentum attribute), so they need to be reset.

        Args:
            state: Environment state. OrderedDict of iterables of sprites.
        """
        pass

    @abc.abstractmethod
    def random_action(self):
        """Sample random action from action space."""
        pass

    @abc.abstractmethod
    def action_spec(self):
        """Get action spec for the output.
        
        Returns:
            dm_env.specs.ArraySpec or nested structure of such.
        """
        pass
