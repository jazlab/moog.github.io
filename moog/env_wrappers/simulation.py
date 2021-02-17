"""Environment wrapper class that supports mental simulation.

This wrapper is useful when running model-based RL algorithms, because it
provides a ground-truth model of the environment.

Note: If there is stochasticity in the environment (e.g. maybe your physics or
game rules have randomness), this will only take samples at each simulation
step, not representing the full distribution of possible simulations. You can
step and pop many times to estimate the distribution of possibilities.
"""

import copy
from moog import env_wrappers


class SimulationEnvironment(env_wrappers.AbstractEnvironmentWrapper):
    """Environment class supporting mental simulation.
    
    This wrapper supports the normal environment interface, but adds two
    additional methods:
        * sim_step(action): This takes a step in mental simulation, advancing
            the state and returning a timestep. However, it also keeps a
            representation of the current state in a stack, to allow for
            restoring later.
        * sim_pop(index=-1): This restores to a previous state in the mental
            simulation stack. Specifically, it restores to the state at level
            index. For example, if we've called sim_step() two times, then
            sim_pop(-1) or sim_pop(1) would restore to the state after the first
            sim_step() call, whereas sim_pop(0) would restore the original state
            before both sim_step() calls.
    
    You can think of sim_step() and sim_pop() as allowing traversals through a
    tree of mental simulation.
    """
    
    def __init__(self, environment):
        """Constructor.

        Args:
            environment: Instance of ../moog/environment.Environment.
        """
        super(SimulationEnvironment, self).__init__(environment)

    def reset(self):
        self.stack = []
        return super(SimulationEnvironment, self).reset()

    def step(self, action):
        """Step the environment with an action."""
        if self.stack:
            self.sim_pop(index=0)
        self.stack = []
        return super(SimulationEnvironment, self).step(action)
    
    def sim_step(self, action):
        """Take a simulation step of the environment with an action."""
        if self._environment.reset_next_step:
            # Should not simulate across episode boundaries
            return None
        self.stack.append({
            'state': copy.deepcopy(self._environment.state),
            'meta_state': copy.deepcopy(self._environment.meta_state),
            'action_space': copy.deepcopy(self._environment.action_space),
            'physics': copy.deepcopy(self._environment.physics),
            'game_rules': copy.deepcopy(self._environment.game_rules),
            'task': copy.deepcopy(self._environment.task),
            'step_count': copy.deepcopy(self._environment.step_count),
            'reset_next_step': copy.deepcopy(self._environment.reset_next_step),
        })
        return super(SimulationEnvironment, self).step(action)
    
    def sim_pop(self, index=-1):
        """Pop and restore the state at index off the stack."""
        restore_data = self.stack[index]
        self._environment._state = restore_data['state']
        self._environment.action_space = restore_data['action_space']
        self._environment.physics = restore_data['physics']
        self._environment.game_rules = restore_data['game_rules']
        self._environment.task = restore_data['task']
        self._environment._meta_state = restore_data['meta_state']  #pylint: disable=protected-access
        self._environment.step_count = restore_data['step_count']
        self._environment._reset_next_step = restore_data['reset_next_step']  #pylint: disable=protected-access
        self.stack = self.stack[:index]
