---
title: 'Modular Object-Oriented Games: A Task Framework for Reinforcement Learning, Psychology, and Neuroscience'
tags:
  - Python
  - AI
  - Reinforcement Learning
  - environment
  - objects
  - psychology
  - psychophysics
  - neurophysiology

authors:
  - name: Nick Watters
    orcid: 0000-0002-7757-7700
    affiliation: 1
  - name: Joshua Tenenbaum
    orcid: 0000-0002-1925-2035
    affiliation: 1, 2
  - name: Mehrdad Jazayeri
    orcid: 0000-0002-9764-6961
    affiliation: 1, 3
affiliations:
 - name: Department of Brain and Cognitive Sciences, Massachusetts Institute of Technology
   index: 1
 - name: Center for Brains, Minds and Machines, Massachusetts Institute of Technology
   index: 2
 - name: McGovern Institute of Brain Research, Massachusetts Institute of Technology
   index: 3
date: 17 February 2021
bibliography: paper.bib
---

# Introduction

In recent years, trends towards studying object-based games have gained momentum
in the fields of artificial intelligence, cognitive science, psychology, and
neuroscience. In artificial intelligence, interactive physical games are now a
common testbed for reinforcement learning [@sutton2018reinforcement;
@mnih2013playing] and object representations are of particular interest for
sample efficient and generalizable AI [@greff2020binding; @van2019perspective;
@lake2017building]. In cognitive science and psychology, object-based games are
used to study a variety of cognitive capacities, such as planning, intuitive
physics, and intuitive psychology [@ullman2017mind]. Developmental psychologists
also use object-based visual stimuli to probe questions about object-oriented
reasoning in infants and young animals [@spelke2007core; @wood2020reverse]. In
neuroscience, object-based computer games have recently been used to study
decision-making and physical reasoning in both human and non-human primates
[@fischer2016functional; @mcdonald2019bayesian; @yoo2020neural,
@Rajalingham2021].

Furthermore, a growing number of researchers are studying tasks using a
combination of approaches from these fields. Comparing artificial agents with
humans or animals performing the same tasks can help constrain models of
human/animal behavior, generate hypotheses for neural mechanisms,
and may ultimately facilitate building more intelligence artificial agents
[@willke2019comparison; @hassabis2017neuroscience].

However, building a task that can be played by AI agents, humans, and animals is
a time-consuming undertaking because existing platforms are typically designed
for only one of these purposes. Professional game engines are designed for human
play and are often heavy-weight libraries that are difficult to customize for
training AI agents and animals. Reinforcement learning platforms are designed
for AI agents but are often too slow or inflexible for neuroscience work.
Existing psychology and neurophysiology platforms are too limited to easily
support complex interactive games.

In this work we offer a solution, a game engine that is highly customizable and
designed to support tasks that can be played by AI agents, humans, and animals.

# Summary

The [``Modular Object-Oriented Games``
(``MOOG``)](https://github.com/jazlab/moog.github.io) library is a
general-purpose python-based platform for interactive games. It aims to satisfy
the following criteria:

* Usable for both reinforment learning and psychology, with [DeepMind
  dm_env](https://github.com/deepmind/dm_env) and [OpenAI
  Gym](https://gym.openai.com/) [@openai_gym] interfaces for RL agents and an
  [MWorks](https://mworks.github.io/) interface for psychology and
  neurophysiology.
* Highly customizable. Environment physics, reward structure, agent interface,
  and more can be customized.
* Easy to rapidly prototype tasks. Tasks can be composed in a single short file.
* Light-weight and efficient. Most tasks run quickly, almost always faster than
  100 frames per second on CPU and often much faster than that.
* Facilitates procedural generation for randomizing task conditions every trial.


# Intended Users

MOOG was designed for use by the following kinds of researchers:

* Machine learning researchers studying reinforcement learning in
  2.5-dimensional physical environments who want to quickly implement tasks in
  Python.
* Psychology researchers who want more flexibility than existing psychology
  platforms afford.
* Neurophysiology researchers who want to study interactive games yet still need
  to precisely control stimulus timing.
* Machine learning researchers studying unsupervised learning, particularly in
  the video domain. MOOG can be used to procedurally generate video datasets
  with controlled statistics.

MOOG may be particularly useful for interdisciplinary researchers studying AI
agents, humans, and animals (or some subset thereof) all playing the same task.


# Design

The core philosophy of MOOG is **"one task, one file."** Namely, a task should
be implemented with a single configuration file. This configuration file is a
short "recipe" for the task, containing as little substantive code as possible,
and should define a set of components to pass to the MOOG environment. See
Figure 1 for a schematic of these components.

![Components of a MOOG environment. See main text for
details.](env_schematic.png)

A MOOG environment receives the following components (or callables returning
them) from the configuration file:

* **State**. The state is a collection of sprites. Sprites are polygonal shapes
  with color and physical attributes (position, velocity, angular velocity, and
  mass). Sprites are 2-dimensional, and the state is 2.5-dimensional with
  z-ordering for occlusion. The initial state can be procedurally generated from
  a custom distribution at the beginning of each episode. The state is
  structured in terms of layers, which helps hierarchical organization. See
  [state_initialization](https://github.com/jazlab/moog.github.io/tree/master/moog/state_initialization)
  for procedural generation tools.
* **Physics**. The physics component is a collection of forces that operate on
  the sprites. There are a variety of forces built into MOOG (collisions,
  friction, gravity, rigid tethers, ...) and additional custom forces can also
  be used. Forces perturb the velocity and angular velocity of sprites, and the
  sprite positions and angles are updated with Newton's method. See
  [physics](https://github.com/jazlab/moog.github.io/tree/master/moog/physics)
  for more.
* **Task**. The task defines the rewards and specifies when to terminate a
  trial. See
  [tasks](https://github.com/jazlab/moog.github.io/tree/master/moog/tasks) for
  more.
* **Action Space**. The action space allows the subject to control the
  environment. Every environment step calls for an action from the subject.
  Action spaces may impart a force on a sprite (like a joystick), move a sprite
  in a grid (like an arrow-key interface), set the position of a sprite (like a
  touch-screen), or be customized. The action space may also be a composite of
  constituent action spaces, allowing for multi-agent tasks and multi-controller
  games. See
  [action_spaces](https://github.com/jazlab/moog.github.io/tree/master/moog/action_spaces)
  for more.
* **Observers**. Observers transform the environment state into a observation
  for the subject/agent playing the task. Typically, the observer includes a
  renderer producing an image. However, it is possible to implement a custom
  observer that exposes any function of the environment state. The environment
  can also have multiple observers. See
  [observers](https://github.com/jazlab/moog.github.io/tree/master/moog/observers)
  for more.
* **Game Rules** (optional). If provided, the game rules define dynamics or
  transitions not captured by the physics. A variety of game rules are provided,
  including rules to modify sprites when they come in contact, conditionally
  create new sprites, and control phase structure of trials (e.g. fixation phase
  to stimulus phase to response phase). See
  [game_rules](https://github.com/jazlab/moog.github.io/tree/master/moog/game_rules)
  for more.

Importantly, all of these components can be fully customized. If a user would
like a physics force, action space, or game rule not provided by MOOG, they can
implement a custom one, inheriting from the abstract base class for that
component. This can typically be done with only a few lines of code.

The modularity of MOOG facilitates code re-use across experimental paradigms.
For example, if a user would like to both collect behavior data from humans
using a continuous joystick and train RL agents with discrete action spaces on
the same task, they can re-use all other components in the task configuration,
only changing the action space.

For users interested in doing psychology or neurophysiology, we include an
example of how to run MOOG through [MWorks](https://mworks.github.io/), a
platform with precise timing control and interfaces for eye trackers, HID
devices, electrophysiology software, and more.


# Example Tasks

![Timelapse images of four example tasks. Left-to-right: (i) Pong - The subject
aims to catch the yellow ball with the green paddle, (ii) Red-Green - The
subject tries to predict whether the blue ball with contact the red square or
the green square, (iii) Pac-Man - The subject moves the green agent to catch
yellow pellets while avoiding the red ghosts, (iv) Collisions - the green agent
avoids touching the bouncing polygons.](example_tasks.png)

See the [example
configs](https://github.com/jazlab/moog.github.io/tree/master/oog_demos/example_configs)
for a variety of task config files. Four of those are shown in Figure 2. See the
[demo
documentation](https://github.com/jazlab/moog.github.io/tree/master/oog_demos)
for videos of them all and instructions for how to run them with a python gui.


# Limitations

Users should be aware of the following limitations of MOOG before choosing to
use it for their research:

* **Not 3D**. MOOG environments are 2.5-dimensional, meaning that they render in
  2-dimensions with z-ordering for occlusion. MOOG does not support 3D sprites.
* **Very simple graphics**. MOOG sprites are monochromatic polygons. There are
  no textures, shadows, or other visual effects. Composite sprites can be
  implemented by creating multiple overlapping sprites, but still the graphics
  complexity is very limited. This has the benefit of a small and easily
  parameterizable set of factors of variation of the sprites, but does make MOOG
  environments visually unrealistic.
* **Imperfect collisions**. MOOG's collision module implements Newtonian
  rotational mechanics, but it is not as robust as professional physics engines
  (e.g. can be unstable if object are moving very quickly and many collisions
  occur simultaneously).


# Related Software

Professional game engines (e.g. [Unity](https://unity.com/) and
[Unreal](https://www.unrealengine.com/)) and visual reinforcement learning
platforms (e.g. [DeepMind Lab](https://arxiv.org/abs/1612.03801) [@dm_lab],
[Mujoco](http://www.mujoco.org/) [@mujoco], and
[VizDoom](http://vizdoom.cs.put.edu.pl/)) are commonly used in the machine
learning field for task implementation. While MOOG has some limitations compared
to these (see above), it does also offer some advantages:

* **Python**. MOOG tasks are written purely in python, so users who are most
  comfortable with python will find MOOG easy to use.
* **Procedural Generation**. MOOG facilitates procedural generation, with a
  [library of compositional
  distributions](https://github.com/jazlab/moog.github.io/tree/master/moog/state_initialization/distributions.py)
  to randomize conditions across trials.
* **Online Simulation**. MOOG supports online model-based RL, with a [ground
  truth
  simulator](https://github.com/jazlab/moog.github.io/tree/master/moog/env_wrappers/simulation.py)
  for tree search.
* **Psychophysics**. MOOG can be run with MWorks, a psychophysics platform.
* **Speed**. MOOG is fast on CPU. While the speed depends on the task and
  rendering resolution, MOOG typically runs at ~200fps with 512x512 resolution
  on a CPU, which is much faster than one would get with DeepMind Lab or Mujoco
  and at least as fast as Unity and Unreal.

Python-based physics simulators, such as
[PyBullet](https://pybullet.org/wordpress/) [@pybullet] and
[Pymunk](http://www.pymunk.org/en/latest/), are sometimes used in the psychology
literature. While these offer highly accurate collision simulation, MOOG offers
the following advantages:

* **Customization**. Custom forces and game rules can be easily implemented in
  MOOG.
* **Psychophysics, Procedural Generation, and Online Simulation**, as described
  above.
* **RL Interface**. A task implemented in MOOG can be used out-of-the-box to
  train RL agents, since MOOG is python-based and has DeepMind dm_env and OpenAI
  Gym interfaces.

Psychology and neurophysiology researchers often use platforms such as
[PsychoPy](https://www.psychopy.org/) [@psychopy],
[PsychToolbox](http://psychtoolbox.org/) [@psychtoolbox], and
[MWorks](https://mworks.github.io/). These allow precise timing control and
coordination with eye trackers and other controllers. MOOG can interface with
MWorks to leverage all of those features and offers the following additional
benefits:

* **Flexibility**. MOOG offers a large scope of interactive tasks. Existing
  psychophysics platforms are not easily customized for game-like tasks, action
  interfaces, and arbitrary object shapes.
* **Physics**. Existing psychophysics platforms do not have built-in physics,
  such as forces, collisions, etc.
* **RL Interface**, as described above.


# Acknowledgements

We thank Chris Stawarz and Erica Chiu for their contributions to the codebase.
We also thank Ruidong Chen, Setayesh Radkani, and Michael Yoo for their feedback
as early users of MOOG.

# References