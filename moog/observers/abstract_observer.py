"""Abstract observer."""

import abc


class AbstractObserver(abc.ABC):
    """Abstract observer class.
    
    All observers must inherit from this class.
    """
    
    @abc.abstractmethod
    def __call__(self, state):
        """Observe the environment state.
        
        Args:
            state: OrderedDict of iterables of sprites.

        Returns:
            observation. Observation of the environment state. Type and size of
                the observation may vary over time, depending on the observer.
        """
        pass

    @abc.abstractmethod
    def observation_spec(self):
        """Get observation spec for the output.
        
        Returns:
            dm_env.specs.ArraySpec or nested structure of such.
        """
        pass
