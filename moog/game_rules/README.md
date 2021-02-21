# Game Rules

The [MOOG Environment class](../environment.py) takes a `game_rules` keyword
argument that is an iterable of game rules. These game rules implement all of
the dynamics of the environment not implemented by the physics. Intuitively,
physics should implement forces, whereas game rules should implement
higher-level transition dynamics, such as sprites appearing/disappearing,
changing color, etc.

Game rules must satisfy the API of [`AbstractRule`](./abstract_rule.py). In this
directory are a few different kinds of game rules. These can be combined in
various ways to implement a wide variety of tasks (see our [example
tasks](https://github.com/jazlab/moog.github.io/blob/master/moog_demos/README.md)).

However, if game rules are insufficient for you to implement your task, by all
means implement a custom game rule inheriting from `AbstractRule` --- typically
this can be done with only a few lines of code. In the
[`functional_maze.py`](../../moog_demos/example_configs/functional_maze.py)
example config we have an example of a custom game rule implemented in the
config.

## Task phases

We want to draw particular attention to the rules in
[task_phases.py](./task_phases.py), which are useful when your task trials have
phases. For example, your task may have a fixation phase at the beginning of
each trial. The `Phase` and `PhaseSequence` provide a way to compose trial
phases with conditional transitions between them.

For examples of phase rules, see the
[`match_to_sample`](../../moog_demos/example_configs/match_to_sample.py) and
[`multi_tracking_with_feature.py`](../../moog_demos/example_configs/multi_tracking_with_feature.py)
example configs.
