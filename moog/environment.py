"""Environment.

This contains the Environment class. Evey MOOG task is an instance of this
Environment. It inherits from the dm_env.Environment API, but see
env_wrappers.gym_wrapper for an OpenAI Gym interface.
"""

import dm_env


class Environment(dm_env.Environment):
    """Environment class.

    Every object-oriented game is an instance of this class. This class pulls
    together all of the components of the game, such as the state initializer,
    physics (physics and forces), task (reward function), action space,
    observer (renderer), and game rules (game_rules). This is the only place
    where all of those components interact.

    The state of the environment is an OrderedDict containing all of physical
    objects (sprites) in the environment. The keys of the state are strings, and
    the values are iterables of sprite.Sprite instances. This state can be
    thought of as an ordered set of layers in the environment. This state is
    passed into physics and action space for updating, into the task for
    computing reward, and into the observer for rendering.
    """

    def __init__(self,
                 state_initializer,
                 physics,
                 task,
                 action_space,
                 observers,
                 game_rules=(),
                 meta_state_initializer=None):
        """Constructor.

        Args:
            state_initializer: Callable returning an initial state, which is an
                OrderedDict of iterables of sprite.Sprite instances. This is
                called at the beginning of each episode.
            physics: Instance of physics.AbstractPhysics. Must have methods:
                * reset(state)
                * step(state) --- in-place update the state each timestep.
            task: Instance of tasks.AbstractTask. Must have methods:
                * reset(state, meta_state)
                * reward(state, meta_state, step_count) returning scalar reward
                    and bool should_reset.
            action_space: Instance of action_spaces.AbstractActionSpace. Must
                have methods:
                * step(state, action)
                * reset(state)
            observers: Dict. Each value must be an instance of
                observers.AbstractObserver, hence callable taking in state and
                returning observation. The keys are the keys of an observation.
                For example, if you would like the environments' observations
                (returned in the timestep of each step) to contain multiple
                kinds of observations (e.g. a rendered image and a state
                description), you can let this observers argument be a
                dictionary {key_0: observer_0, key_1: observer_1} and the
                observation will be a dictionary {key_0: obs_0, key_1: obs_1}.
            game_rules: Iterable of instances of
                game_rules.AbstractRule. Each element is called on the state
                and meta_state every environment step and can modify them
                in-place.
            meta_state_initializer: Optional callable returning environment
                meta_state. If provided, is called every episode reset.
                Environment meta_state is only used by task and game rules.
        """
        self.state_initializer = state_initializer
        self.physics = physics
        self.task = task
        self.action_space = action_space
        self.observers = observers
        self.game_rules = game_rules

        if meta_state_initializer is None:
            self._meta_state_initializer = lambda: None
        else:
            self._meta_state_initializer = meta_state_initializer

    def reset(self):
        """Reset (start a new episode)."""
        self._reset_next_step = False
        self.step_count = 0
        
        self._state = self.state_initializer()
        self._meta_state = self._meta_state_initializer()
        self.task.reset(self._state, self._meta_state)
        self.physics.reset(self._state)
        self.action_space.reset(self._state)
        for rule in self.game_rules:
            rule.reset(self._state, self._meta_state)
            rule.step(self._state, self._meta_state)
        
        return dm_env.restart(self.observation())

    def step(self, action):
        """Step the environment with an action."""
        if self._reset_next_step:
            return self.reset()

        # Apply the game_rules
        for rule in self.game_rules:
            rule.step(self._state, self._meta_state)

        # Apply the action
        self.action_space.step(self._state, action)

        # Step the physics
        self.physics.step(self._state)

        # Compute reward
        self.step_count += 1
        reward, should_reset = self.task.reward(
            self._state, self._meta_state, self.step_count)

        # Take observation
        observation = self.observation()

        # Return transition
        if should_reset:
            self._reset_next_step = True
            return dm_env.termination(reward=reward, observation=observation)
        else:
            return dm_env.transition(reward=reward, observation=observation)

    def observation(self):
        """Returns a dictionary of observations."""
        return {k: observer(self._state)
                for k, observer in self.observers.items()}

    def observation_spec(self):
        """Returns a dictionary of dm_env specs for the observations."""
        observation_specs = {
            name: observer.observation_spec()
            for name, observer in self.observers.items()
        }
        return observation_specs

    def action_spec(self):
        """Returns the action space's .action_spec()."""
        return self.action_space.action_spec()

    @property
    def state(self):
        """State of environment."""
        return self._state
    
    @property
    def meta_state(self):
        """Meta-state of environment."""
        return self._meta_state

    @property
    def reset_next_step(self):
        """Whether to reset (start a new episode) on the next step."""
        return self._reset_next_step
