"""Composite task."""

from . import abstract_task
import numpy as np


class CompositeTask(abstract_task.AbstractTask):
    """CompositeTask task.

    This combines multiple tasks at once, summing the rewards from each of them.
    This can be useful for example to have a predator/prey task where there are
    positive rewards for catching the prey and negative rewards for being caught
    by the predators.
    """

    def __init__(self, *tasks, timeout_steps=np.inf):
        """Constructor.

        Args:
            tasks: Tasks to compose. Reward will be the sum of the rewards from
                each of these tasks.
            timeout_steps: After this number of steps since reset, a reset is
                forced.
        """
        self._tasks = tasks
        self._timeout_steps = timeout_steps

    def reset(self, state, meta_state):
        for task in self._tasks:
            task.reset(state, meta_state)

    def reward(self, state, meta_state, step_count):
        """Compute reward."""
        reward = 0
        should_reset = step_count >= self._timeout_steps
        for task in self._tasks:
            task_reward, task_should_reset = task.reward(
                state, meta_state, step_count)
            reward += task_reward
            should_reset = should_reset or task_should_reset
        
        return reward, should_reset
