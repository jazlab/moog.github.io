# Modular Object-Oriented Games

<img src="../readme_visuals/moog_header.gif" width="900">

## Description

The core philosophy of MOOG is **"one task, one file."** Namely, each task can
be implemented with a single configuration file. This configuration file should
be a short "recipe" for the task, containing as little substantive code as
possible, and should define a set of components to pass to the MOOG environment.
See [configs](../moog_demos/example_configs) for example configuration files.

A MOOG [environment](./environment.py) has the following components. It receives
these (or callables returning these) from the configuration file:

* **State**. The state is a collection of [sprites](./sprite.py). Sprites are
  polygonal shapes with color and physical attributes (position, velocity,
  angular velocity, and mass). Sprites are 2-dimensional, and the state is
  2.5-dimensional with z-ordering for occlusion. The initial state can be
  procedurally generated from a custom distribution at the beginning of each
  episode. The state is structured in terms of layers, which helps hierarchical
  organization. See [state_initialization](./state_initialization) for
  procedural generation tools.
* **Physics**. The physics is a collection of forces that operate on the
  sprites. There are a variety of forces built into MOOG (collisions, friction,
  gravity, rigid tethers, ...) and it is easy to implement your own custom
  force. Forces perturb the velocity and angular velocity of sprites, and the
  sprite positions and angles are updated with Newton's method. See
  [physics](./physics) for more.
* **Task**. The task defines the rewards and specifies when to terminate a
  trial. See [tasks](./tasks) for more.
* **Action Space**. The action space allows the subject to control the
  environment. Every environment step calls for an action from the subject.
  Action spaces may impart a force on a sprite (like a joystick), move a sprite
  in a grid (like an arrow-key interface), set the position of a sprite (like a
  touch-screen), or be customized. The action space may also be a composite of
  constituent action spaces, allowing for multi-agent tasks and multi-controller
  games. See [action_spaces](./action_spaces) for more.
* **Observers**. Observers expose an observation of the environment at each
  timestep, allowing subjects to observe the environment state. Typically, the
  observer includes a renderer producing an image. However, it is possible to
  implement a custom observer that exposes any function of the environment
  state. The environment can also have multiple observers. See
  [observers](./observers) for more.
* **Game Rules** (optional). If provided, the game rules define dynamics of the
  environment not captured by the physics. A variety of game rules are provided,
  including rules to modify sprites when they come in contact, create new
  sprites upon a condition, and control phase structure of trials (e.g. fixation
  phase --> stimulus phase --> response phase). You can also implement your own
  custom game rules (as in the
  [`functional_maze`](../moog_demos/example_configs/functional_maze.py) example
  config). See [game_rules](./game_rules) for more.
* **Meta-state** (optional). If provided, the meta-state contains addition state
  information not represented by the sprites. It is often unused, but can be
  used to record the phase of the task or the game level (this can be useful for
  analysis, particularly of psychophysics data). It is fully customizable.

## Examples

The best way to become familiar with the MOOG framework is by looking at example
task configuration files. See the [`moog_demos` README](../moog_demos/README.md)
for videos of example configs and the
[`moog_demos/example_configs`](../moog_demos/example_configs) directory for the
example config files. The simplest config is
[`predators_arena`](../moog_demos/example_configs/predators_arena.py), so that
is a good place to start.

## Environment

The [environment.py](./environment.py) file contains the `Environment` class.
All MOOG environments are instances of this class. See the [website
documentation](https://jazlab.github.io/moog.github.io/moog/environment.html)
for details about all methods.

Environment functionality can be modified or augmented with wrappers. The
[`env_wrappers`](./env_wrappers) directory contains some example, including
wrappers for multi-agent play, logging, OpenAI Gym interface, and a wrapper for
model-based RL with ground truth simulation. See the [env_wrappers
documentation](https://jazlab.github.io/moog.github.io/moog/env_wrappers/index.html)
for details.

## Sprite

The [sprite.py](./sprite.py) file contains the `Sprite` class. The state of a
MOOG environment is an OrderedDict of iterables of Sprite instances. A sprite
has factors [`x`, `y`, `shape`, `angle`, `scale`, `aspect_ratio`, `c0`, `c1`,
`c2`, `opacity`, `x_vel`, `y_vel`, `angle_vel`, `mass`, `metadata`]. The
sprite's constructor takes these (or some subset of them) as keyword arguments.
These factors may be modified by various components, such as physics and game
rules, and can always be accessed as attributes of the sprite. See the [sprite
documentation](https://jazlab.github.io/moog.github.io/moog/sprite.html#moog.sprite.Sprite)
for details.
