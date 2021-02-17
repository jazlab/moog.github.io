"""Rule that creates new sprites."""

from . import abstract_rule
import itertools


class CreateSprites(abstract_rule.AbstractRule):
    """Create new sprites and add them to a specified layer."""

    def __init__(self, layer, generator, without_overlapping=()):
        """Constructor.

        Args:
            layer: String. Must be a key in the environment state. Layer to
                which appearing sprites are added.
            generator: Callable returning a list of sprites. Must have
                'without_overlapping' kwarg. See
                ../state_initialization/sprite_generators.generate_sprites().
            without_overlapping: Iterable of strings. Each element must be a key
                in the environment state. Created sprites will not overlap
                sprites in these layers.
        """
        self._layer = layer
        self._generator = generator
        self._without_overlapping = without_overlapping

    def step(self, state, meta_state):
        """Apply rule to state."""
        del meta_state
        
        without_overlapping = list(itertools.chain(
            *[state[k] for k in self._without_overlapping]))
        new_sprite = self._generator(without_overlapping=without_overlapping)
        state[self._layer].extend(new_sprite)
