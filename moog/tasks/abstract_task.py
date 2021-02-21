"""Abstract task."""

import abc


class AbstractTask(abc.ABC):
    """Abstract task class.
    
    All tasks must inherit from this class.
    """

    @abc.abstractmethod
    def reset(self, state, meta_state):
        """Reset the task.

        This should do whatever the task needs to do to itself when the
        environment resets. For example, this could reset a count of the number
        of steps taken in the episode, zero out any running state information,
        etc.
        
        Args:
            state: OrderedDict of iterables of sprites.
        """
        pass
    
    @abc.abstractmethod
    def reward(self, state, meta_state, step_count):
        """Get reward and whether to reset based on the environment state.
        
        Args:
            state: OrderedDict of iterables of sprites.
            step_count: Current number of environment steps since last reset.

        Returns:
            reward: Scalar. Reward for current state.
            should_reset: Bool. Whether environment should reset on next step.
        """
        pass
