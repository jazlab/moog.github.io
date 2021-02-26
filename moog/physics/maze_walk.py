"""Maze walk classes.

The classes in this file can be used to walk sprites in mazes, either randomly
or with a deterministic policy. They should be used as forces in a
.physics.Physics instance.
"""

import abc
import numpy as np
from moog import physics
from moog import maze_lib

# Small float to snap positions to grid
_EPSILON = 1e-5


class AbstractMazeWalk(physics.AbstractForce, metaclass=abc.ABCMeta):
    """Abstract maze walk class.
    
    All maze walk classes should inherit from this.
    """

    def __init__(self, speed, maze_layer='walls'):
        """Constructor.

        Args:
            speed: Constant speed at which sprites walk.
            maze_layer: String. Layer in the environment state containing the
                maze wall sprites.
        """
        self._speed = speed
        self._maze_layer = maze_layer

    def reset(self, state):
        """Resetting re-infers the maze from the state."""
        self._maze = maze_lib.Maze.from_state(
            state, maze_layer=self._maze_layer)

    def step(self, *sprites, updates_per_env_step=1):
        """Step the sprites, updating their velocities."""
        for sprite in sprites:
            self._step_sprite(sprite, updates_per_env_step=updates_per_env_step)

    @abc.abstractmethod
    def _step_sprite(self, sprite, updates_per_env_step=1):
        """Step a sprite, updating its velocity."""
        pass

    def _get_pos_vel(self, sprite, updates_per_env_step=1):
        """Get position, velocity, and whether entering intersection.
        
        Args:
            sprite: Sprite instance.
            updates_per_env_step: Int. Number of physics steps per environment
                step.

        Returns:
            position: Position of the sprite.
            velocity: Velocity of the sprite, adjusted to have self._speed
                speed if nonzero.
            entering_intersection: Bool. Whether or not the sprite is entering
                an intersection in the next step.
        """
        position = sprite.position
        velocity = self._speed * np.sign(sprite.velocity)
        next_position = position + velocity / updates_per_env_step

        intersection = (
            self._maze.grid_side * self._get_nearest_point(position) +
            self._maze.half_grid_side)

        dist_next_current = sum(np.abs(next_position - position))
        dist_intersection_next = sum(np.abs(next_position - intersection))
        dist_intersection_current = sum(np.abs(next_position - intersection))

        entering_intersection = (
            dist_next_current > dist_intersection_current and 
            dist_next_current > dist_intersection_next)

        return position, velocity, entering_intersection

    def _get_nearest_point(self, position):
        """Get nearest point of the maze to a position.
        
        Args:
            position: Float array of size (2,).

        Returns:
            nearest_inds: Int array of size (2,). Indices of the nearest maze
                point to position.
        """
        nearest_inds = np.round(position / self._maze.grid_side - 0.5)
        return nearest_inds.astype(int)


class RandomMazeWalk(AbstractMazeWalk):
    """Random maze walk."""

    def __init__(self, speed, maze_layer='walls', prevent_backtracking=True,
                 allow_wall_backtracking=False, only_turn_at_wall=False):
        """Constructor.
        
        Applying this physics to sprites makes them walk with constant speed in
        a maze, taking random turns at corners and intersections.

        Args:
            speed: Float. Speed for the sprite to move at.
            maze_layer: String. Layer in the environment state containing the
                maze sprites.
            prevent_backtracking: Bool. Whether to prevent backtracking
                (changing direction to go the opposite way).
            allow_wall_backtracking: Bool. Whether to allow backtracking if the
                sprite cannot go forward. If False, the sprite will turn when it
                hits a wall but never go back the way it came.
            only_turn_at_wall: Bool. Whether to only turn when cannot continue
                straight.
        """
        super(RandomMazeWalk, self).__init__(speed, maze_layer=maze_layer)
        self._prevent_backtracking = prevent_backtracking
        self._allow_wall_backtracking = allow_wall_backtracking
        self._only_turn_at_wall = only_turn_at_wall

    def _update_valid_directions(self, valid_directions, velocity):
        """Update valid directions based on what kinds of backtracking to allow.

        This updating occurs in place.

        Args:
            valid_directions: Binary array of shape (2, 2) indicating which
                cardinal directions are available to move in.
            velocity: Current sprite velocity.
        """
        # If not preventing backtracking, all open directions are valid
        if not self._prevent_backtracking:
            return
        axis = np.argmax(np.abs(velocity))
        direction = np.sign(velocity[axis])

        # If velocity is zero, all open directions are valid
        if direction == 0:
            return
        
        # If hit a wall and allow wall backtracking, all open directions are
        # valid
        can_continue = valid_directions[axis, int(0.5 * (1 + direction))]
        if not can_continue and self._allow_wall_backtracking:
            return
        # If not hit a wall and only turn at wall, then continue
        if can_continue and self._only_turn_at_wall:
            valid_directions.fill(0)
            valid_directions[axis, int(0.5 * (1 + direction))] = 1
            return

        # If none of the above conditions are true, prevent backtracking
        valid_directions[axis, int(0.5 * (1 - direction))] = False
    
    def _step_sprite(self, sprite, updates_per_env_step=1):
        """Update a sprite's velocity.
        
        Args:
            sprite: Sprite instance.
            updates_per_env_step: Int. Number of physics steps per environment
                step.
        """
        if np.isinf(sprite.mass):
            return

        position, velocity, entering_intersection = self._get_pos_vel(
            sprite, updates_per_env_step=updates_per_env_step)
        nearest_inds = self._get_nearest_point(position)
        
        # If sprite is entering an intersection or stationary, find the valid
        # directions to move in.
        if entering_intersection:
            valid_directions = self._maze.valid_directions(
                nearest_inds[0], nearest_inds[1])
            self._update_valid_directions(valid_directions, velocity)
        elif np.all(velocity == 0.):
            rounded_position = (
                self._maze.half_grid_side + nearest_inds * self._maze.grid_side)
            on_grid = np.abs(rounded_position - position) < _EPSILON
            if np.all(on_grid):
                valid_directions = self._maze.valid_directions(
                    nearest_inds[0], nearest_inds[1])
            else:
                valid_directions = np.zeros((2, 2))
                valid_directions[1 - np.argmax(on_grid)] = 1
        else:
            sprite.velocity = velocity
            return
        
        # Sample new direction to move in
        sample = valid_directions * np.random.rand(2, 2)
        sample_ind = np.argmax(np.ravel(sample))

        # Update velocity to move in new direction, but don't eliminate current
        # velocity as that might be needed to get us to the intersection
        velocity[sample_ind // 2] = (
            (1 + _EPSILON) * self._speed * (2 * (sample_ind % 2) - 1))
        sprite.velocity = velocity


class DeterministicMazeWalk(AbstractMazeWalk):
    """Deterministic maze walk."""

    def __init__(self, speed, step_velocities, maze_layer='walls'):
        """Constructor.

        Args:
            speed: Float. Speed for the sprite to move at.
            step_velocities: Iterable of 2-iterables. Each element is the
                perscribed velocity of a sprite at a timestep. The elements are
                read out front-to back each time a sprite enters an
                intersection. Note that to handle multiple sprites you have to
                interleave their velocities, since this class has no way of
                knowing which sprite it is stepping.
            maze_layer: String. Layer in the environment state containing the
                maze sprites.
        """
        super(DeterministicMazeWalk, self).__init__(
            speed, maze_layer=maze_layer)
        self._step_velocities = [np.array(v) for v in step_velocities]

    def _step_sprite(self, sprite, updates_per_env_step=1):
        """Update a sprite's velocity.
        
        Args:
            sprite: Sprite instance.
            updates_per_env_step: Int. Number of physics steps per environment
                step.
        """
        _, velocity, entering_intersection = self._get_pos_vel(
            sprite, updates_per_env_step=updates_per_env_step)

        if entering_intersection or np.all(velocity == 0.):
            # If self._step_velocities is not empty use the next one. Otherwise,
            # do nothing.
            if len(self._step_velocities) > 0:
                new_velocity = self._step_velocities.pop(0)
                if np.any(np.sign(new_velocity) != np.sign(velocity)):
                    sprite.velocity = np.clip(
                        (1 - _EPSILON) * velocity + new_velocity, -self._speed,
                        -self._speed)
