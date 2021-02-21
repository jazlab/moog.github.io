# State initialization

The [MOOG Environment class](../environment.py) takes a `state_initializer`
keyword argument that is a callable returning an initial state at the beginning
of each episode. This directory contains tools for creating such a state
initializer.

## Distributions

The [`distributions.py`](./distributions.py) file contains a library of
distributions that can be composed to create arbitrarily complex distributions
of sprite factors. If you find yourself beginning to implement a complicate
sampler to generate sprite factors, check
[`distributions.py`](./distributions.py) to see if some of the tools there can
be used.

## Sprite generators

The [`sprite_generators.py`](./sprite_generators.py) file contains functions for
generating lists of multiple sprites from the same distribution. In many cases,
these functions don't save much space in the config (it's easy to sample by
hand: `sprites = [Sprite(**my_factor_distrib.sample() for _ in
range(num_sprites)]`). However, the tools in
[`sprite_generators.py`](./sprite_generators.py) do offer some additional
functionality that may be useful:
* Sampling disjoint sprites. The `disjoint` boolean argument in a sprite
  generator can be used to force all generated sprites to not overlap.
* Sampling sprites that don't overlap other sprites. The `without_overlapping`
  argument can be a list of sprites, in which case all generated sprites are
  forced to not overlap any of those in the given list.

## Cross-Trial Dependencies

In some cases, you might want to implement a task where there the initial state
is not sampled independently each trial, but instead depends on some aspect of
previous trials (e.g. maybe there's a Poisson rule switch that happens every few
trials). The easiest way to do this is to make the `state_initializer` a method
of a class. Namely, instead of making it a function in the config, make it a
method of a class and let the class's state monitor the necessary cross-trial
data. For a simple example of this, see the
[`predators_arena.py`](../../moog_demos/example_configs/predators_arena.py)
example config.
