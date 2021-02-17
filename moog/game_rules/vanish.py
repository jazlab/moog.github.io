"""Rules that make sprites vanish."""

from . import abstract_rule
from .contact_rules import get_contact_indices
import abc
import numpy as np


class Vanish(abstract_rule.AbstractRule, metaclass=abc.ABCMeta):
    """Metaclass for rules that make sprites vanish."""

    def __init__(self, layer):
        """Constructor.

        Args:
            layer: String. Must be a key in the environment state. Sprites in
                this layer will be removed from the state if they are indexes by
                self._get_vanish_inds().
        """
        self._layer = layer

    @abc.abstractmethod
    def _get_vanish_inds(self, state):
        """Takes in state and returns an iterable of indices.
        
        Returned indices must index which elements of state[self._layer] should
            vanish.
        """
        pass

    def step(self, state, meta_state):
        """Remove sprites in specified indices of vanishing layer."""
        del meta_state
        
        vanish_inds = self._get_vanish_inds(state)
        count_vanished_already = 0
        for i in vanish_inds:
            state[self._layer].pop(i - count_vanished_already)
            count_vanished_already += 1


class VanishByFilter(Vanish):
    """Makes a sprite vanish based on a boolean function of itself."""

    def __init__(self, layer, filter_fn=None):
        """Constructor.

        Args:
            layer: String. Must be a key in the environment state. Sprites in
                this layer will be removed from the state if filter_fn returns
                True on them.
            filter_fn: None or Function taking in a sprite and return a bool
                indicating whether the given sprite should vanish. If None,
                defaults to True, i.e. vanishing all sprites in layer.
        """
        super(VanishByFilter, self).__init__(layer)
        if filter_fn is None:
            self._filter_fn = lambda _: True
        else:
            self._filter_fn = filter_fn

    def _get_vanish_inds(self, state):
        vanish_inds = np.argwhere(
            [self._filter_fn(s) for s in state[self._layer]])[:, 0]
        return vanish_inds


class VanishOnContact(Vanish):
    """Makes a sprite vanish if it is in contact with another sprite."""

    def __init__(self, vanishing_layer, contacting_layer):
        """Constructor.

        Args:
            vanishing_layer: String. Must be a key in the environment state.
                Sprites in this layer will be removed from the state if they
                contact a sprite in contacting_layer.
            contacting_layer: String. Must be a key in the environment state.
        """
        super(VanishOnContact, self).__init__(vanishing_layer)
        self._get_contact_indices = get_contact_indices(
            vanishing_layer, contacting_layer)

    def _get_vanish_inds(self, state):
        contact_inds = self._get_contact_indices(state)
        return np.unique([i for (i, j) in contact_inds])
