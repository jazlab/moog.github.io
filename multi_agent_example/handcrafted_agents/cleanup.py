"""Handcrafted agents for the cleanup task.

These are some simple hand-crafted agents that are useful for playing against in
the demo.

These are intended for example purposes only, and are necessary to play the demo
on ../configs/cleanup.py.
"""

import numpy as np


class RandomAgent():
    """Agent takes random action in [0, 1] x [0, 1], regardless of the state."""

    def __init__(self, name):
        self._name = name

    def step(self, observation):
        del observation
        return np.random.uniform(-1., 1., size=(2,))


def _target_homing(targets, agent):
    """Return the action that moves agent towards the nearest target.
    
    If targets is empty, return the zero-action.
    """
    if not targets:
        return np.zeros(2)
    targets_positions = [s.position for s in targets]
    targets_diffs = [pos - agent.position for pos in targets_positions]
    nearest_target_ind = np.argmin(np.linalg.norm(targets_diffs, axis=1))
    nearest_target_diff = targets_diffs[nearest_target_ind]
    action = nearest_target_diff / np.linalg.norm(nearest_target_diff)
    return action


def _ripe_fruit_homing(state, agent_name):
    """Get action towards the nearest ripe fruit in the cleanup task.."""
    agent_sprite = state[agent_name][0]
    fruits = state['fruits']
    ripe_fruits = list(filter(lambda s: s.c2 > 0.6, fruits))
    return _target_homing(ripe_fruits, agent_sprite)


def _poison_fountain_homing(state, agent_name):
    """Get action towards the nearest poisonous fountain in the cleanup task."""
    agent_sprite = state[agent_name][0]
    fruits = state['fountains']
    poison_fountains = list(filter(lambda s: s.c2 < 0.6, fruits))
    return _target_homing(poison_fountains, agent_sprite)


class SelfishAgent():
    """This agent is a free-rider, only collecting ripe fruit."""

    def __init__(self, name):
        self._name = name

    def step(self, observation):
        return _ripe_fruit_homing(observation['state'], self._name)


class SelflessAgent():
    """This agent only cleans poisonous fountains and never collects fruit."""

    def __init__(self, name):
        self._name = name

    def step(self, observation):
        return _poison_fountain_homing(observation['state'], self._name)


class FickleAgent():
    """This agent switches regularly between a given set of strategies."""

    def __init__(self, *agents, steps_per_agent=25):
        """Constructor.

        Example usage to produce an agent that switches between being selfish
        and being selfless every 100 steps:
            ```python
            FickleAgent(
                SelfishAgent(name='agent_layer_name'),
                SelflessAgent(name='agent_layer_name'),
                steps_per_agent=100)
            ```

        Args:
            agents: Iterable of agents. These are the agents that this
                FickleAgent mimics.
        """
        self._agents = agents
        self._num_agents = len(self._agents)
        self._steps_per_agent = steps_per_agent
        self._current_agent = 1
        self.switch()

    def switch(self):
        self._steps_until_switch = self._steps_per_agent
        self._current_agent = (self._current_agent + 1) % self._num_agents

    def step(self, observation):
        if self._steps_until_switch == 0:
            self.switch()
        self._steps_until_switch -= 1
        return self._agents[self._current_agent].step(observation)
