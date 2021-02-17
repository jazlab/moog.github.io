"""Rules that re-center a sprite periodically.

Typically used in first-person games to keep the agent in the middle of the
screen and move all non-agent sprites when the agent moves to preserve their
position relative to the agent.
"""

from . import abstract_rule
import numpy as np


class KeepNearCenter(abstract_rule.AbstractRule):
    """Translates many sprites at once to re-center an agent.
    
    This is used in first-person games, where there is a background grid that
    does not extend to infinity so all the other sprites must periodically be
    snapped back near the center once the agent reaches the grid size away.
    """

    def __init__(self,
                 agent_layer,
                 layers_to_center,
                 grid_x,
                 grid_y=None):
        """Constructor.

        Args:
            agent_layer: String. Must be a key in the environment state. The
            first sprite in state[agent_layer] will be kept near [0.5, 0.5].
            layers_to_center: Iterable of strings. Elements must be keys in
                environment state. All sprites in these layers will be centered
                along with the agent.
            grid_x: Maximum x-distance from 0.5 allowed by the agent before
                centering. Recentering adjusts x-positions by exactly this
                amount.
            grid_x: Maximum y-distance from 0.5 allowed by the agent before
                centering. Recentering adjusts y-positions by exactly this
                amount.
        """
        self._agent_layer = agent_layer

        if agent_layer not in set(layers_to_center):
            layers_to_center = list(layers_to_center) + [agent_layer]
        self._layers_to_center = layers_to_center
    
        if grid_y is None:
            grid_y = grid_x
        self._grid_cell = np.array([grid_x, grid_y])

    def _move_sprites(self, state, delta_pos):
        for layer in self._layers_to_center:
            for sprite in state[layer]:
                sprite.position = sprite.position + delta_pos

    def step(self, state, meta_state):
        """Run the rule, centering layers if necessary."""
        del meta_state
        
        agent = state[self._agent_layer][0]
        agent_pos = agent.position - np.array([0.5, 0.5])
        delta_pos = (
            -1. * self._grid_cell * (agent_pos > self._grid_cell) + 
            self._grid_cell * (agent_pos < -1. * self._grid_cell)
        )

        if any(delta_pos):
            self._move_sprites(state, delta_pos)
