"""Portal.

The entrypoint is Portal, a rule that teleports a sprite from one portal sprite
position to another.
"""

from . import abstract_rule
import numpy as np


class Portal(abstract_rule.AbstractRule):
    """Makes a sprite teleport if it enters a portal sprite."""

    def __init__(self, teleporting_layer, portal_layer):
        """Constructor.

        The environment state must have an even number of sprites in
        portal_layer, because portals are paired up in order. I.e. if there are
        4 portals, the first two will teleport to each other and the second two
        will teleport to each other.

        Also, once a sprite has teleported, it cannot immediately teleport again
        until it exits the portal sprite. This is kept track of by
        self._currently_teleporting, and is necessary to prevent a sprite from
        immediately teleporting back and forth between portals.

        Args:
            teleporting_layer: String. Must be a key in the environment state.
                Sprites in this layer will be teleported from the state if their
                position enters a sprite in portal_layer.
            portal_layer: String. Must be a key in the environment state.
        """        
        self._teleporting_layer = teleporting_layer
        self._portal_layer = portal_layer

    def reset(self, state, meta_state):
        del state
        del meta_state
        self._currently_teleporting = set()

    def step(self, state, meta_state):
        """Apply rule to state.

        Teleport sprites in teleporting layer if they have entered a portal
        sprite, unless they are in self._currently_teleporting.
        """
        del meta_state

        portals = state[self._portal_layer]
        num_portals = len(portals)
        if num_portals % 2 != 0:
            raise ValueError(
                'There must be an even number of portals, but you have {} '
                'portals.'.format(num_portals))
        
        for sprite in state[self._teleporting_layer]:
            in_portals = [portal.contains_point(sprite.position)
                          for portal in portals]
            in_portal_inds = np.argwhere(in_portals)[:, 0]

            if len(in_portal_inds) == 0:
                # Sprite is not in any portal, so make sure we don't think
                # sprite is currently teleporting
                self._currently_teleporting.discard(sprite.id)
                continue

            if sprite.id in self._currently_teleporting:
                # To prevent immediately teleporting back and forth between
                # portals
                continue
            
            # Teleport the sprite
            entry_ind = in_portal_inds[0]
            exit_ind = entry_ind - 1 if entry_ind % 2 else entry_ind + 1
            sprite.position = np.copy(portals[exit_ind].position)
            self._currently_teleporting.add(sprite.id)
