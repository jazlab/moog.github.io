"""Rules and functions involving sprite contacts.

This file contains the ModifyOnContact game rule, which modifies sprites if they
come in contact.

It also contains the get_contact_counter() function, which can be useful for
counting how many contacts there are between layers, e.g. for use in a
ConditionalRule.
"""

from . import abstract_rule
import itertools


def get_contact_indices(layer_0, layer_1):
    """Get counter that finds index pairs of all contacts between layers.
    
    Args:
        layer_0: String. Must be key in state.
        layer_1: String. Must be key in state.

    Returns:
        _call: Function state --> list, where the elements of the returned list
            are tuples (i_0, i_0) of indices of sprites in layer_0 and layer_1
            that are contacting.
    """

    def _call(state):
        """Gets all (i_0, i_1) such that layer_0[i_0] contacts layer_1[i_1]."""
        contact_indices = []
        for i_0, sprite_0 in enumerate(state[layer_0]):
            for i_1, sprite_1 in enumerate(state[layer_1]):
                if sprite_0.overlaps_sprite(sprite_1):
                    contact_indices.append((i_0, i_1))
        return contact_indices
    return _call


def get_contact_counter(layer_0, layer_1):
    """Get counter that finds how many contacts there are between layers.
    
    Args:
        layer_0: String. Must be key in state.
        layer_1: String. Must be key in state.

    Returns:
        Function state --> int returning how many contacts there are between
            layer_0 and layer_1.
    """
    _get_contact_indices = get_contact_indices(layer_0, layer_1)
    return lambda state: len(_get_contact_indices(state))


class ModifyOnContact(abstract_rule.AbstractRule):
    """Modify sprites if they contact each other."""

    def __init__(self,
                 layers_0,
                 layers_1,
                 modifier_0=None,
                 modifier_1=None,
                 filter_0=None,
                 filter_1=None,):
        """Constructor.

        Applies modifier_0 to those sprites in layers_0 that satisfy filter_0
        and contact a sprite in layers_1. Similarly, applies modifier_1 to those
        sprites in layers_1 that satisfy filter_1 and contact a sprite in
        layers_0.
        
        Args:
            layers_0: String or iterable of strings. Must be layer name(s) in
                environment state.
            layers_1: String or iterable of strings. Must be layer name(s) in
                environment state.
            modifier_0: Function taking in a sprite and modifying in place.
            modifier_1: Function taking in a sprite and modifying in place.
            filter_0: Function taking in a sprite and returning bool.
            filter_1: Function taking in a sprite and returning bool.
        """
        if not isinstance(layers_0, (list, tuple)):
            layers_0 = (layers_0,)
        self._layers_0 = layers_0
        if not isinstance(layers_1, (list, tuple)):
            layers_1 = (layers_1,)
        self._layers_1 = layers_1
        
        self._modifier_0 = modifier_0
        self._modifier_1 = modifier_1

        if filter_0 is None:
            filter_0 = lambda x: True
        self._filter_0 = filter_0
        if filter_1 is None:
            filter_1 = lambda x: True
        self._filter_1 = filter_1

    def _modify_asymmetric(self, sprites_modifying, sprites_contacting,
                           modifier, filter):
        """Modify sprites_modifying if contcating sprites_contacting."""
        if modifier is not None:
            for s in sprites_modifying:
                if not filter(s):
                    continue
                contacts = [s.overlaps_sprite(x) for x in sprites_contacting
                            if id(x) != id(s)]
                if any(contacts):
                    modifier(s)

    def step(self, state, meta_state):
        """Apply rule to state."""
        del meta_state
        
        sprites_0 = list(itertools.chain(*[state[k] for k in self._layers_0]))
        sprites_1 = list(itertools.chain(*[state[k] for k in self._layers_1]))
        
        self._modify_asymmetric(
            sprites_0, sprites_1, self._modifier_0, self._filter_0)
        self._modify_asymmetric(
            sprites_1, sprites_0, self._modifier_1, self._filter_1)
