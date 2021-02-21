# Modular Object-Oriented Games

<img src="readme_visuals/moog_header.gif" width="900">

## Description

[Project Website](https://jazlab.github.io/moog.github.io/index.html)

### Summary

This [Modular Object-Oriented Games (MOOG)
library](https://github.com/jazlab/moog.github.io) is a general-purpose
python-based platform for interactive games. It aims to satisfy the following
criteria:

* Highly customizable. Environment physics, reward structure, agent interface,
  and more are customizable.
* Easy to rapidly prototype tasks. Tasks can be composed in a single short file.
* Usable for both reinforment learning and psychology, with [DeepMind
  dm_env](https://github.com/deepmind/dm_env) and [OpenAI
  Gym](https://gym.openai.com/) interfaces for RL agents and an
  [MWorks](https://mworks.github.io/) interface for psychology and
  neurophysiology.
* Light-weight and efficient. Most tasks run quickly, almost always faster than
  100 frames per second on CPU and often much faster than that.
* Facilitates procedural generation for randomizing task conditions every trial.

See [`moog_demos`](moog_demos) for a variety of example tasks.

### Intended Users

MOOG may be useful for the following kinds of researchers:

* Machine learning researchers studying reinforcement learning in
  2.5-dimensional (2-dimensional with occlusion) physical environments who want
  to quickly implement tasks without having to wrangle with more complicated
  game engines that aren't designed for RL.
* Psychology researchers who want more flexibility than existing psychology
  platforms afford.
* Neurophysiology researchers who want to study interactive games yet still need
  to precisely control stimulus timing.
* Machine learning researchers studying unsupervised learning, particularly in
  the video domain. MOOG can be used to procedurally generate video datasets
  with controlled statistics.

### Introduction

The core philosophy of MOOG is **"one task, one file."** Namely, each task can
be implemented with a single configuration file. This configuration file should
be a short "recipe" for the task, containing as little substantive code as
possible, and should define a set of components to pass to the MOOG environment.
See [the MOOG README](moog/README.md) for more details.

We also include an example [MWorks](https://mworks.github.io/) interface for
running psychophysics experiments, as well as a [python demo
script](moog_demos/run_demo.py) for testing task prototypes.

### Features Compared to Existing Platforms

Compared to professional game engines (Unity, Unreal, etc.) and existing visual
reinforcement learning platforms (DM-Lab, Mujoco, VizDoom, etc.):

* **Python**. MOOG tasks are written purely in python, so users who are most
  comfortable with python will find MOOG easy to use.
* **Procedural Generation**. MOOG facilitates procedural generation, with a
  [library of compositional
  distributions](moog/state_initialization/distributions.py) to randomize
  conditions across trials.
* **Online Simulation**. MOOG supports online model-based RL, with a [ground
  truth simulator](moog/env_wrappers/simulation.py) for tree search.
* **Psychophysics**. MOOG can be run with MWorks, a psychophysics platform.
* **Speed**. MOOG is fast on CPU. While the speed depends on the task and
  rendering resolution, MOOG typically runs at ~200fps with 512x512 resolution
  on a CPU, which is faster than one would get with DM-Lab or Mujoco and at
  least as fast as Unity and Unreal.

Compared to existing python game platforms (PyBullet, Pymunk, etc.):

* **Customization**. Custom forces and game rules can be easily implemented in
  MOOG.
* **Psychophysics, Procedural Generation, and Online Simulation**, as described
  above.
* **RL Interface**. A task implemented in MOOG can be used out-of-the-box to
  train RL agents, since MOOG is python-based and has DeepMind dm_env and OpenAI
  Gym interfaces.

Compared to existing psychophysics platforms (PsychoPy, PsychToolbox, MWorks):

* **Flexibility**. MOOG offers a large scope of interactive tasks. Existing
  psychophysics platforms are not easily customized for game-like tasks, action
  interfaces, and arbitrary object shapes.
* **Physics**. Existing psychophysics platforms do not have built-in physics,
  such as forces, collisions, etc.
* **RL Interface**, as described above.

MOOG can [interface with MWorks](mworks), allowing users to leverage the MOOG
task framework while also allowing for precise timing control and interfaces
with eye-trackers, joysticks, and electrophysiology software.


### Limitations

* **Not 3D**. MOOG environments are 2.5-dimensional, meaning that they render in
  2-dimensions with z-ordering for occlusion. MOOG does not support 3D sprites.
* **Very simple graphics**. MOOG sprites are monochromatic polygons. There are
  no textures, shadows, or other visual effects. Composite sprites can be
  implemented by creating multiple overlapping sprites, but still the graphics
  complexity is very limited. This has the benefit of a small and easily
  parameterizable set of factors of variation of the sprites, but does make MOOG
  environments visually unrealistic.
* **Imperfect physics**. MOOG's physics engine is simple. It uses Newton's
  method to effect action-at-a-distance forces. MOOG does include a collision
  module that implements rotational mechanics, but it is not as robust as more
  professional physics engines and can have instabilities (particularly if
  multiple objects collide simultaneously). See
  [`moog_demos/example_configs/falling_balls.py`](moog_demos/example_configs/falling_balls.py)
  for an extreme example of unstable physics.


## Getting Started

See the [project website](https://jazlab.github.io/moog.github.io/index.html)
for API documentation about every file and function in MOOG.

### Installation

If you would like to install this library as a package, you can install using
pip:
```bash
pip install moog-games
```

This will install `moog` and `moog_demos` packages. Be sure to use python 3.7 or
later.

### Running The Demo

Tasks can be played by running the [run_demo](moog_demos/run_demo.py) script, in
which the `--config` flag indicates the task config to demo. For example, to
demo the [pong](moog_demos/example_configs/pong.py) task, you would run:

```bash
python3 -m moog_demos.run_demo --config='moog_demos.example_configs.pong'
```

When this command is run, the demo will produce an interactive display. At the
top of the display is the rendered environment state, in the middle of the
display is a histogram of recent rewards, and at the bottom of the display is a
top-down view of a cartoon joystick. You can click and drag the joystick around
to control the agent avatar. The demo can be terminated by pressing `escape`.

The pong task looks like this:

<img src="readme_visuals/pong.gif" width="300">

You can change the config flag to point to any of the [example
configs](moog_demos/example_configs). They will all run except for `cleanup`,
which is multi-agent so cannot be played by a single-agent demo, (though see
[`multi_agent_example`](multi_agent_example/) for more about that).

### Implementing Tasks

Before implementing your own tasks, please read [the MOOG
README](moog/README.md).

To begin implementing your own task, we recommend first looking at all the
example configs in [`moog_demos`](moog_demos) and copying one with some
similarities to your task into a working directory. Then modify it incrementally
to your specification.

To demo your config, copy [`run_demo.py`](moog_demos/run_demo.py) into your
working directory and run it with
```bash
$ python3 run_demo.py --config='path.to.your.config' --level=$your_config_level
```

## Contact and Support

Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for information about support.
Please email Nick Watters at nwatters@mit.edu with questions and feedback.

## Reference

Some parts of this codebase are derived from
[Spriteworld](https://github.com/deepmind/spriteworld/). See the Spriteworld
license in [LICENSE-spriteworld](LICENSE-spriteworld).
