"""SetPosition action space for controlling agent avatars.

This action space sets the position of an agent. This is commonly used with a
touch-screen or mouse-click interface.
"""

from . import abstract_action_space
from dm_env import specs
import numpy as np


class SetPosition(abstract_action_space.AbstractActionSpace):
    """SetPosition action space."""

    def __init__(self, action_layers='agent', inertia=0.):
        """Constructor.
        
        Args:
            agent_layer: String or iterable of strings. Elements (or itself if
                string) must be keys in the environment state. All sprites in
                these layers will be acted upon by this action space.
            inertia: Float in [0, 1]. Inertia of the action layer sprite
                positions. Zero means position is set instantly to action value,
                and 1 means the action has no effect on sprite positions.
        """
        if not isinstance(action_layers, (list, tuple)):
            action_layers = (action_layers,)
        self._action_layers = action_layers
        self._inertia = inertia

        self._action_spec = specs.BoundedArray(
            shape=(2,), dtype=np.float32, minimum=0, maximum=1)

    def step(self, state, action):
        """Apply action to environment state.

        Args:
            state: Ordereddict. Environment state.
            action: Numpy float array of size (2), in [0, 1]. Position to set
                for the agent(s) in self._action_layers.
        """
        for action_layer in self._action_layers:
            for sprite in state[action_layer]:
                sprite.position = (
                    self._inertia * sprite.position +
                    (1 - self._inertia) * action
                )

    def random_action(self):
        """Return randomly sampled action."""
        return np.random.uniform(0., 1., size=(2,))
            
    def action_spec(self):
        return self._action_spec
