# Action Spaces

Every MOOG Environment has an action space. This action space is given to the
[environment's
constructor](https://jazlab.github.io/moog.github.io/moog/environment.html#moog.environment.Environment),
and must satisfy the API of
[`AbstractActionSpace`](https://jazlab.github.io/moog.github.io/moog/action_spaces/abstract_action_space.html).

In this directory are a few example action spaces for common interfaces. In
particular, it contains:

* [`Grid`](https://jazlab.github.io/moog.github.io/moog/action_spaces/grid.html).
  This is an [up, down, left, right, donothing] action space commonly used for
  grid-world environments.
* [`Joystick`](https://jazlab.github.io/moog.github.io/moog/action_spaces/joystick.html).
  This is a continuous 2-dimensional ([0, 1] x [0, 1]) joystick action space,
  commonly used to impart a continuous force (or velocity) to move an agent.
* [`SetPosition`](https://jazlab.github.io/moog.github.io/moog/action_spaces/set_position.html).
  This is a continuous 2-dimensional ([0, 1] x [0, 1]) action space that sets
  the position of a sprite. It can be used to simulate a touch-screen interface
  (e.g. by controlling the position of a perhaps transparent sprite that serves
  as a finger).
* [`Composite`](https://jazlab.github.io/moog.github.io/moog/action_spaces/composite.html).
  This provides a way to compose action spaces for multiple interfaces or
  multiple agents. For example, if your experiment involves joystick motion and
  eye-tracking, you could have a composite action space with a `Joystick` and a
  `SetPosition` component, where the `SetPosition` controls a transparent sprite
  that serves as a marker of eye position and the `Joystick` controls an agent
  avatar.
