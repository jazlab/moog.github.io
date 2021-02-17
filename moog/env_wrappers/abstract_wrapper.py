"""Abstract wrapper.

This file contains AbstractEnvironmentWrapper, and abstract base class for
environment wrappers that mimics the interface of the underlying environment.

One might ask why not make the environment wrappers inherit from
../environment.Environment instead of wrapping and exposing all methods and
properties. The reason is because users may want to use multiple environment
wrappers in arbitrary combinations. If we had used inheritance we would not have
been able to accomodate these combinations of wrappers.

One might also ask why not use mixins. The reason is again to support using
multiple wrappers. Some of the wrappers override the same functions (e.g.
.__init__(), .step()), so disentangling them to operate as multiple mixins would
not be possible without sacrificing code cleanness/simplicity.
"""

import abc


class AbstractEnvironmentWrapper(abc.ABC):
    """Abstract environment wrapper class.
    
    All environment wrappers must inherit from this class.
    """

    def __init__(self, environment):
        self._environment = environment

    def reset(self):
        return self._environment.reset()
    
    def step(self, action):
        return self._environment.step(action)

    def observation(self):
        return self._environment.observation()

    def observation_spec(self):
        return self._environment.observation_spec()

    def action_spec(self):
        return self._environment.action_spec()
    
    @property
    def state(self):
        return self._environment.state

    @property
    def meta_state(self):
        return self._environment.meta_state

    @property
    def state_initializer(self):
        return self._environment.state_initializer

    @property
    def physics(self):
        return self._environment.physics

    @property
    def task(self):
        return self._environment.task

    @property
    def action_space(self):
        return self._environment.action_space

    @property
    def observers(self):
        return self._environment.observers

    @property
    def game_rules(self):
        return self._environment.game_rules
    
    @property
    def environment(self):
        return self._environment
    
    @property
    def step_count(self):
        return self._environment.step_count

    @property
    def reset_next_step(self):
        return self._environment.reset_next_step
    