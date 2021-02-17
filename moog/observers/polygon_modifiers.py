"""Polygon modifier classes.

These classes are used to modify polygons (sprite vertex arrays, colors, and
opacities), typically before rendering. They can be used to adjust polygon
positions to render first-person, duplicate sprites when simulating torus
geometry, etc.
"""

import abc
import numpy as np


class AbstractPolygonModifier(abc.ABC):
    """Abstract polygon modifier class.
    
    All polygon modifiers must inherit from this class.
    """
    
    @abc.abstractmethod
    def __call__(self, state):
        """Get polygon modifier.
        
        Args:
            state: Environment state.

        Returns:
            Function taking in list of polygons and returning list of polygons.
        """
        pass


class DoNothing(AbstractPolygonModifier):
    """Does not modify polygons."""
    
    def __call__(self, state):
        def _do_nothing(_, sprite):
            return [(sprite.vertices, sprite.color, sprite.opacity)]
        return _do_nothing


class FirstPersonAgent(AbstractPolygonModifier):
    """Modifies polygon positions to put an agent sprite at [0.5, 0.5]."""

    def __init__(self, agent_layer):
        """Constructor.

        Args:
            first_person_agent: String. Will apply a translation such that the
                agent sprite (first sprite in this layer) is at position
                [0.5, 0.5].
        """
        self._agent_layer = agent_layer

    def __call__(self, state):
        """Get polygon modifier that translates lists of polygons."""
        agents = state[self._agent_layer]
        agent_position = agents[0].position
        delta_position = np.array([0.5, 0.5]) - agent_position
        def _sprite_to_polygons(layer, sprite):
            outs = [
                (sprite.vertices + delta_position, sprite.color, sprite.opacity)
            ]
            return outs
        return _sprite_to_polygons


class TorusGeometry(AbstractPolygonModifier):
    """Duplicates polygons at a grid of positions.
    
    Specifically, duplicates polygons at a 3x3 grid of positions centered around
    [0, 0] + polygon_position. This is used in torus environments to make a
    sprite appear to smoothly and incrementally disappear off one edge of the
    arena and reappear on the opposite edge.
    """

    def __init__(self, wrap_layers):
        """Constructor.

        Args:
            wrap_layers: String or iterable of strings. All sprites in these
                layers will be rendered as if the arena is a torus.
        """
        if not isinstance(wrap_layers, (list, tuple)):
            wrap_layers = [wrap_layers]
        self._wrap_layers = wrap_layers

    def __call__(self, state):
        """Get polygon modifier rendering sprites as if the arena is a torus."""
        del state
        def _sprite_to_polygons(layer, sprite):
            squared_polygons = [
                (sprite.vertices + np.array([i, j]), sprite.color,
                 sprite.opacity)
                for i in [-1., 0., 1.]
                for j in [-1., 0., 1.]
            ]
            return squared_polygons
        return _sprite_to_polygons
