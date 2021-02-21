"""Discrete grid action space for controlling agent avatars."""

from . import abstract_action_space
from dm_env import specs
import numpy as np


class Grid(abstract_action_space.AbstractActionSpace):
    """Discrete grid action space.
    
    This action space has 5 actions {left, right, up, down, do-nothing}. These
    actions control either the force or the velocity of the agent(s).
    """

    _ACTIONS = (
        np.array([-1, 0]),  # Move left
        np.array([1, 0]),  # Move right
        np.array([0, -1]),  # Move down
        np.array([0, 1]),  # Move up
        np.array([0, 0]),  # Do not move
    )

    def __init__(self, scaling_factor=1., action_layers='agent',
                 control_velocity=False, momentum=0.):
        """Constructor.
        
        Args:
            scaling_factor: Scalar. Scaling factor multiplied to the action.
            agent_layer: String or iterable of strings. Elements (or itself if
                string) must be keys in the environment state. All sprites in
                these layers will be acted upon by this action space.
            control_velocity: Bool. Whether to control velocity (True) or force
                (False).
            momentum: Float in [0, 1]. Discount factor for previous action. This
                should be zero if control_velocity is False, because imparting
                forces automatically gives momentum to the agent(s) being
                controlled. If control_velocity is True, setting this greater
                than zero gives the controlled agent(s) momentum. However, the
                velocity is clipped at scaling_factor, so the agent only retains
                momentum when stopping or changing direction and does not
                accelerate.
        """
        self._scaling_factor = scaling_factor
        if not isinstance(action_layers, (list, tuple)):
            action_layers = (action_layers,)
        self._action_layers = action_layers
        self._control_velocity = control_velocity
        self._momentum = momentum

        self._action_spec = specs.DiscreteArray(len(self._ACTIONS))

    def step(self, state, action):
        """Apply action to environment state.

        Args:
            state: Ordereddict of layers of sprites. Environment state.
            action: Numpy float array of size (2). Force to apply.
        """
        self._action *= self._momentum

        self._action += Grid._ACTIONS[action]
        self._action = np.clip(
            self._action, -self._scaling_factor, self._scaling_factor)
            
        for action_layer in self._action_layers:
            for sprite in state[action_layer]:
                if self._control_velocity:
                    sprite.velocity = self._action / sprite.mass
                else:
                    sprite.velocity += self._action / sprite.mass

    def reset(self, state):
        """Reset action space at start of new episode."""
        del state
        self._action = np.zeros(2)

    def random_action(self):
        """Return randomly sampled action."""
        return np.random.randint(len(Grid._ACTIONS))
    
    def action_spec(self):
        return self._action_spec
