"""Rules that move sprites from one layer to another."""

from . import abstract_rule
import numpy as np


class ChangeLayer(abstract_rule.AbstractRule):
    """ChangeLayer rule.
    
    This rule is used to change the layer of a sprite based on a filter of the
    sprite. For example, it could be used to change a sprite's layer if the
    sprite has changed to a new color.
    """

    def __init__(self, old_layer, new_layer, filter_fn=None):
        """Constructor.

        Args:
            old_layer: String. Must be a key in the environment state. Sprites
                in this layer will be considered for the layer change.
            new_layer: String. Must be a key in the environment state. Layer to
                which sprites in old_layer may be moved.
            filter_fn: Function sprite -> bool. Whether to move the sprite from
                layer old_layer to layer new_layer. Defaults to always True.
        """
        self._old_layer = old_layer
        self._new_layer = new_layer
        
        if filter_fn is None:
            self._filter_fn =  lambda s: True
        else:
            self._filter_fn = filter_fn

    def step(self, state, meta_state):
        """Apply rule, potential moving sprites from old to new layer."""
        del meta_state
        
        should_change = [self._filter_fn(s) for s in state[self._old_layer]]
        change_inds = np.argwhere(should_change)[:, 0]

        count_changed_already = 0
        for i in change_inds:
            state[self._new_layer].append(
                state[self._old_layer].pop(i - count_changed_already))
            count_changed_already += 1
