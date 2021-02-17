"""Composite action space that composes multiple action spaces.

This is often used for multi-agent games and for games where the subject gives
multiple sources of control, e.g. eye position and joystick movement.
"""

from . import abstract_action_space


class Composite(abstract_action_space.AbstractActionSpace):
    """Composite action space.
    
    Example usage:
        joystick_action_space = maze_action_space.Joystick(
            scaling_factor=0.1, action_layers='agent')
        eye_action_space = action_spaces.SetPosition(
            action_layers='eye_sprite')
        
        action_space = action_spaces.Composite(
            joystick=joystick_action_space,
            eye=eye_action_space,
        )

        # Playing the game...
        action = {'joystick': [0.1, -0.2], 'eye': [0.4, 0.8]}
        env.step(action=action)  # goes to action_space.step(state, action)
    """

    def __init__(self, **action_spaces):
        """Constructor.
        
        Args:
            action_spaces: Dict. Keys are strings and values are action spaces.
                An action is a dictionary with the same set of keys.
        """
        self.action_spaces = action_spaces
        self._action_keys = action_spaces.keys()

        self._action_spec = {
            key: value.action_spec() for key,value in self.action_spaces.items()
        }

    def step(self, state, action):
        """Apply action to environment state.

        Args:
            state: Ordereddict of layers of sprites. Environment state.
            action: Dict. Keys much be the same as self._action_keys. Each value
                will be fed into the action space of the corresponding key.
        """
        for k, v in action.items():
            self.action_spaces[k].step(state, v)

    def reset(self, state):
        for k in self.action_spaces:
            self.action_spaces[k].reset(state)

    def random_action(self):
        """Return randomly sampled action."""
        random_action = {
            k: self.action_spaces[k].random_action() for k in self._action_keys}
        return random_action

    def action_spec(self):
        return self._action_spec

    @property
    def action_keys(self):
        return list(self._action_keys)
