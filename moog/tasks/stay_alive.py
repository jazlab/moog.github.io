"""Task that gives reward at regular intervals during task."""

from . import abstract_task


class StayAlive(abstract_task.AbstractTask):
    """StayAlive task. In this task a reward is given at regular intervals."""

    def __init__(self, reward_period, reward_value=1.):
        """Constructor.

        Args:
            reward_period: Int. Number of steps between each reward.
            reward_value: Scalar. Value of reward given.
        """
        self._reward_period = reward_period
        self._reward_value = reward_value

    def reset(self, state, meta_state):
        pass

    def reward(self, state, meta_state, step_count):
        """Compute reward."""
        del state
        del meta_state
        
        if (step_count + 1) % self._reward_period == 0:
            reward = self._reward_value
        else:
            reward = 0
            
        return reward, False
