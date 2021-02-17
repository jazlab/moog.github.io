"""Rules that modify sprites."""

from . import abstract_rule
import itertools
import numpy as np


class ModifySprites(abstract_rule.AbstractRule):
    """Modify sprites in a layer or set of layers.

    A filter can be applied to modify only sprites within the layer of interest
    that satisfy some condition.
    """

    def __init__(self, layers, modifier, sample_one=False, filter_fn=None):
        """Constructor.

        Args:
            layers: String or iterable of strings. Must be a key (or keys) in
                the environment state. Layer(s) in which sprites are modified.
            modifier: Function taking in a sprite and modifying it in place.
            sample_one: Bool. Whether to sample one sprite to modify if multiple
                satisfy filter_fn at a given step.
            filter_fn: Optional filter function. If specified must take in a
                sprite and return a bool saying whether to consider modifying
                that sprite.
        """
        if isinstance(layers, str):
            layers = [layers]
        self._layers = layers
        self._modifier = modifier
        self._sample_one = sample_one
        self._filter_fn = filter_fn

    def step(self, state, meta_state):
        """Apply rule to state."""
        del meta_state
        
        sprites_to_modify = [s for k in self._layers for s in state[k]]

        if self._filter_fn:
            sprites_to_modify = list(
                filter(self._filter_fn, sprites_to_modify))

        if not sprites_to_modify:
            return

        if self._sample_one:
            sprites_to_modify = [np.random.choice(sprites_to_modify)]

        for sprite in sprites_to_modify:
            self._modifier(sprite)
