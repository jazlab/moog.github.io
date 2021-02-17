"""Task for receiving rewards upon contact."""

from . import abstract_task
import inspect
import numpy as np


class ContactReward(abstract_task.AbstractTask):
    """ContactReward task.
    
    In this task if any sprite in layers_0 contacts any sprite in layers_1, a
    reward is given. Otherwise the reward is zero. Optionally, the task resets
    upon such a contact.

    This can be used for any contact-based reward, such as prey-seeking and
    predator-avoidance.
    """

    def __init__(self,
                 reward_fn,
                 layers_0,
                 layers_1,
                 condition=None,
                 reset_steps_after_contact=np.inf):
        """Constructor.

        Args:
            reward_fn: Scalar or function (sprite_0, sprite_1) --> scalar. If
                function, sprite_0 and sprite_1 are sprites in layers_0 and
                layers_1 respectively.
            layers_0: String or iterable of strings. Reward is given if a sprite
                in this layer(s) contacts a sprite in layers_1.
            layers_1: String or iterable of strings. Reward is given if a sprite
                in this layer(s) contacts a sprite in layers_0.
            condition: Optional condition function. If specified, must have one
                of the following signatures:
                    * sprite_0, sprite_1 --> bool
                    * sprite_0, sprite_1, meta_state --> bool
                The bool is whether to apply reward for those sprites
                contacting.
            reset_steps_after_contact: Int. How many steps after a contact to
                reset the environment. Defaults to infinity, i.e. never
                resetting.
        """
        if not callable(reward_fn):
            self._reward_fn = lambda sprite_0, sprite_1: reward_fn
        else:
            self._reward_fn = reward_fn
        
        if not isinstance(layers_0, (list, tuple)):
            layers_0 = [layers_0]
        self._layers_0 = layers_0

        if not isinstance(layers_1, (list, tuple)):
            layers_1 = [layers_1]
        self._layers_1 = layers_1

        if condition is None:
            self._condition = lambda s_agent, s_target, meta_state: True
        elif len(inspect.signature(condition).parameters.values()) == 2:
            self._condition = lambda s_a, s_t, meta_state: condition(s_a, s_t)
        else:
            self._condition = condition

        self._reset_steps_after_contact = reset_steps_after_contact

    def reset(self, state, meta_state):
        self._steps_until_reset = np.inf

    def reward(self, state, meta_state, step_count):
        """Compute reward.
        
        If any sprite_0 in self._layers_0 overlaps any sprite_1 in
        self._layers_1 and if self._condition(sprite_0, sprite_1, meta_state) is
        True, then the reward is self._reward_fn(sprite_0, sprite_1).

        Args:
            state: OrderedDict of sprites. Environment state.
            meta_state: Environment state. Unconstrained type.
            step_count: Int. Environment step count.

        Returns:
            reward: Scalar reward.
            should_reset: Bool. Whether to reset task.
        """
        reward = 0
        sprites_0 = [s for k in self._layers_0 for s in state[k]]
        sprites_1 = [s for k in self._layers_1 for s in state[k]]
        for s_0 in sprites_0:
            for s_1 in sprites_1:
                if not self._condition(s_0, s_1, meta_state):
                    continue
                if s_0.overlaps_sprite(s_1):
                    reward = self._reward_fn(s_0, s_1)
                    if self._steps_until_reset == np.inf:
                        self._steps_until_reset = (
                            self._reset_steps_after_contact)

        self._steps_until_reset -= 1
        should_reset = self._steps_until_reset < 0

        return reward, should_reset
