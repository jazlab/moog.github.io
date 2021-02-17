# This file was forked and modified from the file here:
# https://github.com/deepmind/spriteworld/blob/master/spriteworld/renderers/pil_renderer.py
# Here is the license header for that file:

# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Python Image Library (PIL/Pillow) renderer."""

from . import abstract_observer
from . import color_maps
from . import polygon_modifiers

from dm_env import specs
import numpy as np
from PIL import Image
from PIL import ImageDraw


class PILRenderer(abstract_observer.AbstractObserver):
    """Render using Python Image Library (PIL/Pillow).
    
    This renders an environment state as an image.
    """

    def __init__(self,
                 image_size=(64, 64),
                 anti_aliasing=1,
                 bg_color=None,
                 color_to_rgb=None,
                 polygon_modifier=None):
        """Construct PIL renderer.

        Args:
            image_size: Int tuple (height, width). Size of output of .render().
            anti_aliasing: Int. Anti-aliasing factor. Linearly scales the size
                of the internal canvas.
            bg_color: None or 3-tuple of ints in [0, 255]. Background color. If
                None, background is (0, 0, 0).
            color_to_rgb: String or Callable converting a tuple (c1, c2, c3) to
                a uint8 tuple (r, g, b) in [0, 255]. If string, must be the name
                of a function in color_maps.py, which will be looked up and
                used.
            polygon_modifier: Instance of
                polygon_modifiers.AbstractPolygonModifier. Callable taking state
                and returning a function converting list of polygons (sprite
                vertex arrays) to another list of polygon vertices. This can be
                used to adjust polygon positions to render first-person,
                duplicate sprites when simulating torus geometry, etc. See
                .polygon_modifiers.py for examples.
        """
        self._image_size = image_size
        self._anti_aliasing = anti_aliasing
        self._canvas_size = (anti_aliasing * image_size[0],
                             anti_aliasing * image_size[1])
        
        if polygon_modifier is None:
            polygon_modifier = polygon_modifiers.DoNothing()
        self._polygon_modifier = polygon_modifier

        if color_to_rgb is None:
            color_to_rgb = lambda x: x
        elif isinstance(color_to_rgb, str):
            color_to_rgb = getattr(color_maps, color_to_rgb)
        self.color_to_rgb = color_to_rgb

        self._observation_spec = specs.Array(
            shape=self._image_size + (3,), dtype=np.uint8)

        if bg_color is None:
            bg_color = (0, 0, 0)
        self._canvas_bg = Image.new('RGB', self._canvas_size, bg_color)

        self._canvas = Image.new('RGB', self._canvas_size)
        self._draw = ImageDraw.Draw(self._canvas, 'RGBA')

    def __call__(self, state):
        """Render sprites.

        The order of layers in the state is background to foreground, and the
        order of sprites within layers is also background to foreground.

        Args:
            state: OrderedDict of iterables of sprites.

        Returns:
            Numpy uint8 RGB array of size self._image_size + (3,).
        """
        self._canvas.paste(self._canvas_bg)
        
        polygon_modifier = self._polygon_modifier(state)
        
        for layer in state:
            for sprite in state[layer]:
                polygons = polygon_modifier(layer, sprite)
                for (vertices, color, opacity) in polygons:
                    vertices = self._canvas_size * vertices
                    color = self.color_to_rgb(color)
                    color = tuple(list(color) + [opacity])
                    self._draw.polygon([tuple(v) for v in vertices], fill=color)
        image = self._canvas.resize(
            self._image_size, resample=Image.ANTIALIAS)

        # PIL uses a coordinate system with the origin (0, 0) at the upper-left,
        # but our environment uses an origin at the bottom-left (i.e.
        # mathematical convention). Hence we need to flip the render vertically
        # to correct for that.
        image = np.flipud(np.array(image))
        
        return image

    @property
    def polygon_modifier(self):
        return self._polygon_modifier

    def observation_spec(self):
        return self._observation_spec
