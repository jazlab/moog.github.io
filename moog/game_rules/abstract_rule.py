"""Abstract game rule."""

import abc


class AbstractRule(abc.ABC):
    """Abstract game rule class.
    
    All game rules must inherit from this class.
    """

    def reset(self, state, meta_state):
        """Reset rule at beginning of every trial.

        This method can be used to reset any attributes of the rule that serve
        as memory within trials.
        
        Args:
            state: OrderedDict of iterables of sprites. Environment state.
            meta_state: meta_state of environment.
        """
        pass
    
    @abc.abstractmethod
    def step(self, state, meta_state):
        """Apply rule to the environment state.

        This method can in-place modify the state however it likes.
        
        Args:
            state: OrderedDict of iterables of sprites. Environment state.
            meta_state: meta_state of environment.
        """
        pass
