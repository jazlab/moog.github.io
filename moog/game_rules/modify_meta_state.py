"""Rules that modify environment meta_state."""

from . import abstract_rule
import itertools
import numpy as np


class ModifyMetaState(abstract_rule.AbstractRule):
    """Modify meta_state.
    
    This can be used for any in-place meta_state modification.
    """

    def __init__(self, modifier):
        """Constructor.

        Args:
            modifier: Function taking in meta_state and modifying it in place.
        """
        self._modifier = modifier

    def step(self, state, meta_state):
        """Apply rule to state, modifying meta_state."""
        del state
        self._modifier(meta_state)


class UpdateMetaStateValue(abstract_rule.AbstractRule):
    """Update a value in the meta_state dictionary.
    
    This rule assumes that the environment meta_state is a dictionary. It is
    commonly used as a one-time rule to indicate a change in task phase.
    """

    def __init__(self, key, value):
        """Constructor.

        Args:
            key: Key of the meta_state dictionary.
            value: Value to set as meta_state[key].
        """
        self._key = key
        self._value = value

    def step(self, state, meta_state):
        """Apply rule to state."""
        del state
        meta_state[self._key] = self._value
