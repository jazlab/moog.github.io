# Physics

The [MOOG Environment class](../environment.py) takes a `physics` keyword
argument that satisfies the API defined in
[`AbstractPhysics`](./abstract_physics.py) and implements forces on and
continuous motion of sprites.

In this directory, the [`physics.py`](./physics.py) file contains a `Physics`
class that composes forces, allows multiple updates per environment step, and is
suitable for most tasks. 

However, for maze-based environments please use the `MazePhysics` class in
[`maze_physics.py`](./maze_physics.py), which constrains sprite motion to a
gridworld maze. See the
[`pacman.py`](../../moog_demos/example_configs/pacman.py) for an example.

In this directory are also a variety of forces. These can be configured and
combined to simulate many sprite dynamics, though, as with all MOOG components,
your are welcome to implment custom forces if those provided are insufficient
for your task. If you do so, be sure to inherit from `AbstractNewtonianForce` if
your force is Newtonian or `AbstractForce` otherwise, both of which are in the
[`abstract_force.py`](./abstract_force.py) file.
