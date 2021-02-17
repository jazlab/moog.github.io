"""Rule that can be used to monitor fixation.

This rule is used to keep track of how long the subject has been fixating on a
sprite (e.g. a fixation cross). The fixation duration is stored in the
environment meta_state, and at each step either incremented if subject is
fixating or reset to 0 otherwise.

A config that uses this rule should have a conditional rule checking whether the
fixation duration in the environment meta_state has reached a threshold, in
order to proceed with the task.
"""

from . import abstract_rule
import numpy as np


class Fixation(abstract_rule.AbstractRule):
    """Fixation rule."""

    def __init__(self,
                 agent_layer,
                 fixation_layer,
                 fixation_threshold=0.1,
                 meta_state_fixation_key='fixation_duration'):
        """Constructor.

        Args:
            agent_layer: String. Key in environment state of agent layer. Agent
                is assumed to be the sprite at index 0 in this layer.
            fixation_layer: String. Key in environment state of agent layer.
                Fixation target is assumed to be the sprite at index 0 in this
                layer.
            fixation_threshold: Float. Distance from agent to fixation target
                considered fixating.
            meta_state_fixation_key: String. Key in meta_state whose value
                contains fixation duration.
        """
        self._agent_layer = agent_layer
        self._fixation_layer = fixation_layer
        self._fixation_threshold = fixation_threshold
        self._meta_state_fixation_key = meta_state_fixation_key

    def reset(self, state, meta_state):
        del state
        meta_state[self._meta_state_fixation_key] = 0

    def step(self, state, meta_state):
        agent = state[self._agent_layer][0]
        target = state[self._fixation_layer][0]
        dist = np.linalg.norm(agent.position - target.position)
        if dist < self._fixation_threshold:
            meta_state[self._meta_state_fixation_key] += 1
        else:
            meta_state[self._meta_state_fixation_key] = 0
