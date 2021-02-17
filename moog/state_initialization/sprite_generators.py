# This file was forked and modified from the file here:
# https://github.com/deepmind/spriteworld/blob/master/spriteworld/sprite_generators.py
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
"""Generators for producing lists of sprites based on factor distributions."""

import itertools
import numpy as np
from moog import sprite


def generate_sprites(factor_dist,
                     num_sprites=1,
                     max_recursion_depth=int(1e4),
                     fail_gracefully=False):
    """Create callable that samples sprites from a factor distribution.

    Example usage:
        ```python
        sprite_factors = distribs.Product(
            [distribs.Continuous('x', 0.2, 0.8),
             distribs.Continuous('y', 0.2, 0.8),
             distribs.Continuous('x_vel', -0.03, 0.03),
             distribs.Continuous('y_vel', -0.03, 0.03)],
            shape='circle, scale=0.1, c0=255, c1=0, c2=0,
        )
        sprite_gen = sprite_generators.generate_sprites(
            sprite_factors, num_sprites=lambda: np.random.randint(3, 6))
        
        def _state_initializer():
            ...
            other_sprites = ...
            ...
            sprites = sprite_gen(
                disjount=True, without_overlapping=other_sprites)
            state = collections.OrderedDict([
                ('other_sprites', other_sprites),
                ('sprites', sprites),
            ])
        ```

    Args:
        factor_dist: The factor distribution from which to sample. Should be an
            instance of spriteworld.factor_distributions.AbstractDistribution.
        num_sprites: Int or callable returning int. Number of sprites to
            generate per call.
        max_recursion_depth: Int. Maximum recursion depth when rejection
            sampling to generate sprites without overlap.
        fail_gracefully: Bool. Whether to return a list of sprites or raise
            RecursionError if max_recursion_depth is exceeded.

    Returns:
        _generate: Callable that returns a list of Sprites.
    """
    def _overlaps(s, other_sprites):
        """Whether s overlaps any sprite in other_sprites."""
        if len(other_sprites) == 0:
            return False
        else:
            overlaps = [s.overlaps_sprite(x) for x in other_sprites]
            return any(overlaps)

    def _generate(disjoint=False, without_overlapping=[]):
        """Return a list of sprites.
        
        Args:
            disjoint: Boolean. If true, all generated sprites will be disjoint.
            without_overlapping: Optional iterable of ../sprite/Sprite
                instances. If specified, all generated sprites will not overlap
                any sprites in without_overlapping.
        """
        n = num_sprites() if callable(num_sprites) else num_sprites
        sprites = []
        for _ in range(n):
            s = sprite.Sprite(**factor_dist.sample())
            count = 0
            while _overlaps(s, without_overlapping):
                if count > max_recursion_depth:
                    if fail_gracefully:
                        return sprites
                    else:
                        raise RecursionError(
                            'max_recursion_depth exceeded trying to initialize '
                            'a non-overlapping sprite.')
                count += 1
                s = sprite.Sprite(**factor_dist.sample())
            sprites.append(s)
            if disjoint:
                without_overlapping = without_overlapping + [s]
        
        return sprites

    return _generate


def chain_generators(*sprite_generators):
    """Chain generators by concatenating output sprite sequences.

    Essentially an 'AND' operation over sprite generators. This is useful when
    one wants to control the number of samples from the modes of a multimodal
    sprite distribution.

    Note that factor_distributions.Mixture provides weighted mixture
    distributions, so chain_generators() is typically only used when one wants
    to forces the different modes to each have a non-zero number of sprites.

    Args:
        *sprite_generators: Callable sprite generators.

    Returns:
        _generate: Callable returning a list of sprites.
    """

    def _generate(*args, **kwargs):
        return list(itertools.chain(*[generator(*args, **kwargs)
                                      for generator in sprite_generators]))

    return _generate


def sample_generator(sprite_generators, p=None):
    """Sample one element from a set of sprite generators.

    Essential an 'OR' operation over sprite generators. This returns a callable
    that samples a generator from sprite_generators and calls it.

    Note that if sprite_generators each return 1 sprite, this functionality can
    be achieved with spriteworld.factor_distributions.Mixture, so
    sample_generator is typically used when sprite_generators each return
    multiple sprites. Effectively it allows dependant sampling from a multimodal
    factor distribution.

    Args:
        sprite_generators: Iterable of callable sprite generators.
        p: Probabilities associated with each generator. If None, assumes
            uniform distribution.

    Returns:
        _generate: Callable sprite generator.
    """

    def _generate(*args, **kwargs):
        sampled_generator = np.random.choice(sprite_generators, p=p)
        return sampled_generator(*args, **kwargs)

    return _generate


def shuffle(sprite_generator):
    """Randomize the order of sprites sample from sprite_generator.

    This is useful because sprites are z-layered with occlusion according to
    their order, so if sprite_generator is the output of chain_generators(),
    then sprites from some component distributions will always be behind sprites
    from others.

    An alternate design would be to let the environment handle sprite ordering,
    but this design is preferable because the order can be controlled more
    finely. For example, this allows the user to specify one sprite (e.g. the
    agent's body) to always be in the foreground while all the others are
    randomly ordered.

    Args:
        sprite_generator: Callable return a list of sprites.

    Returns:
        _generate: Callable sprite generator.
    """

    def _generate(*args, **kwargs):
        sprites = sprite_generator(*args, **kwargs)
        order = np.arange(len(sprites))
        np.random.shuffle(order)
        return [sprites[i] for i in order]

    return _generate
