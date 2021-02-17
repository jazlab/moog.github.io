"""Task for resetting environment upon a condition of the environment state."""

from . import abstract_task
import inspect
import numpy as np


class Reset(abstract_task.AbstractTask):
    """Reset task.
    
    This task resets the environment when (or some fixed number of steps after)
    the environment state meets a condition. It can be used for example to reset
    once all prey are gone.

    There is also an option to specify a reward (as a function of the state) to
    be computed when the condition is met.
    """

    def __init__(self, condition, reward_fn=None, steps_after_condition=np.inf):
        """Constructor.

        Args:
            condition: Function with one of the following signatures:
                    * state --> bool
                    * state, meta_state --> bool
                The bool is whether to reset.
            reward_fn: Reward function taking in state and returning scalar
                reward. Only called if condition(state) is True.
            steps_after_condition: Int. Number of steps after condition is True
                to reset.
        """
        if len(inspect.signature(condition).parameters.values()) == 1:
            self._condition = lambda state, meta_state: condition(state)
        else:
            self._condition = condition
        self._steps_after_condition = steps_after_condition

        if reward_fn is None:
            reward_fn = lambda _: 0.
        self._reward_fn = reward_fn

    def reset(self, state, meta_state):
        # We reset to infinity, because self._steps_until_reset will be
        # decremented every time self.reward() is called, so is only set to a
        # finite value when the condition is met and reset is imminent.
        self._steps_until_reset = np.inf

    def reward(self, state, meta_state, step_count):
        """Compute reward."""
        del step_count
        if (self._steps_until_reset == np.inf and
                self._condition(state, meta_state)):
            reward = self._reward_fn(state)
            self._steps_until_reset = self._steps_after_condition
        else:
            reward = 0.

        self._steps_until_reset -= 1
        should_reset = self._steps_until_reset < 0

        return reward, should_reset
