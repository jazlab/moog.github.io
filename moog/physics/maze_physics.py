"""Maze physics.

The MazePhysics class in this file is a physics object that forces sprites in
specified layers to move on a grid in a maze.

It is typically used as corrective physics for a physics.Physics instance.
"""

import numpy as np
from moog import maze_lib
from moog import physics as physics_lib

# Small imprecision tolerance when determining whether a sprite is on a grid
# line in a maze.
_EPSILON = 1e-5


class MazePhysics(physics_lib.AbstractPhysics):
    """Maze physics class."""

    def __init__(self, maze_layer='walls', avatar_layers=(),
                 constant_speed=None, max_speed=None):
        """Constructor.

        This class constrains sprites in avatar_layers to move in the maze
        specified by the sprites in maze_layer.
        
        Args:
            maze_layer: String. Name of the layer in the environment containing
                the maze wall sprites.
            avatar_layers: Iterable of strings. Sprites in these layers are
                constrained to move in the maze.
            constant_speed. None or float. If Float, all sprites in
                avatar_layers move with this constant speed.
            max_speed: None or float. If float, all sprites in avatar_layers
                move no faster than max_speed. If None, then no speed limit. If
                constant_speed is not None, then max_speed is ignored.
        """
        super(MazePhysics, self).__init__(updates_per_env_step=1)
        self._maze_layer = maze_layer
        self._avatar_layers = avatar_layers
        self._constant_speed = constant_speed
        self._max_speed = max_speed

    def reset(self, state):
        """Resetting re-infers the maze from the state."""
        self._maze = maze_lib.Maze.from_state(
            state, maze_layer=self._maze_layer)

    def _get_position_affordances(self, position):
        """Get affordances of a position.
        
        The affordances of a position indicate how far one can travel from that
        position in each direction.

        Args:
            position: Numpy float array of size (2,).
        
        Returns:
            position: Numpy float array of size (2,). The input position,
                possibly perturbed if any coordinates are within _EPSILON of
                grid lines. This rounding (essentially snapping the position to
                the grid if it's close) ensures that numerical instabilities
                don't accumulate.
            affordances: Numpy float array of size (2, 2). affordances[:, 0]
                indicate how far the position could be moved in the negative
                direction for each axis before hitting an intersection or going
                off the maze. affordances[:, 1] are similiar for the positive
                direction.
        """
        grid_side = self._maze.grid_side
        half_grid_side = self._maze.half_grid_side

        # Figure out which axes are on grid lines, and snap position to grid
        nearest_inds = (np.round(position / grid_side - 0.5)).astype(int)
        rounded_position = half_grid_side + nearest_inds * grid_side
        on_grid = np.abs(rounded_position - position) < _EPSILON
        new_position = np.copy(position)
        new_position[on_grid] = rounded_position[on_grid]

        # Get the affordances
        affordances = np.zeros((2, 2))
        inds = ((position - half_grid_side) // grid_side).astype(int)
        inds[on_grid] = nearest_inds[on_grid]
        if not any(on_grid):  # This should never happen
            raise ValueError(
                'Object is not on the maze grid. This could happen if you '
                'initialized or somehow forced a sprite position to lie off '
                'the maze grid. This could also happen if you are using a '
                'corrective_physics after the MazePhysics, in which case that '
                'later corrective_physics is adjusting the velocities produced '
                'by the MazePhysics and making the sprites run off the maze. '
                'This is bad --- if MazePhysics is used, it must be the last '
                'corrective_physics object in the physics, so that the '
                'velocities it produces are immediately enacted.')
        elif all(on_grid):  # The position lies at a grid vertex
            valid_directions = self._maze.valid_directions(inds[0], inds[1])
            affordances = valid_directions * grid_side * np.array(
                [[-1., 1.], [-1., 1.]])
        else:  # The position lies on a grid edge
            i = 1 - np.argwhere(on_grid)[0][0]
            lower = inds[i] * grid_side + half_grid_side - position[i]
            upper = (inds[i] + 1) * grid_side + half_grid_side - position[i]
            affordances[i] = np.array([lower, upper])

        return position, affordances            

    def _get_new_velocity(self, position, velocity, affordances, axis=None):
        """Get the maze-adjusted velocity.

        Args:
            position: Numpy float array of size (2,).
            velocity: Numpy float array of size (2,).
            affordances: Numpy float array of size (2, 2) containing how far one
                can travel from position in each direction.
            axis: Axis on which to travel. If None, uses the highest-speed axis.
        """
        if axis is None:  # Find the highest-speed axis
            axis = np.argmax(np.abs(velocity))
            
        if affordances[axis, 0] <= velocity[axis] <= affordances[axis, 1]:
            # Can travel with velocity along axis without hitting walls or
            # intersections
            velocity[1 - axis] = 0
            return velocity
        else:
            direction = int(0.5 + 0.5 * np.sign(velocity[axis]))
            if affordances[axis, direction] == 0:
                # We cannot move in the axis direction, so resort to other axis
                axis = 1 - axis
                direction = int(0.5 + 0.5 * np.sign(velocity[axis]))
                if affordances[axis, direction] == 0 or velocity[axis] == 0:
                    # Cannot move anywhere
                    return np.zeros(2)
                else:  # Move along other axis
                    return self._get_new_velocity(
                        position, velocity, affordances, axis=axis)
            else:  # Affordances let us reach a vertex
                # Compute the affordances at the vertex
                position[axis] += affordances[axis, direction]
                _, vertex_affordances = self._get_position_affordances(position)

                # Compute the remaining velocity at the vertex
                scaling = affordances[axis, direction] / velocity[axis]
                velocity_remainder = (1. - scaling) * np.copy(velocity)

                # Compute the velocity after hitting the vertex
                velocity_post_vertex = self._get_new_velocity(
                    position, velocity_remainder, vertex_affordances)
                
                # Update velocity to travel to the vertex then add the
                # post-vertex velocity
                velocity *= scaling
                velocity[1 - axis] = 0
                velocity += velocity_post_vertex

                return velocity

    def _update_sprite_angle(self, sprite, new_velocity):
        """Update sprite angle if necessary.
        
        This makes sprites rotate as they take turns in a maze.
        """
        # We have a little case-work to do here, because of the possibility of
        # some velocity components being zero
        if new_velocity[0] == 0 and new_velocity[1] == 0:
            new_angle = np.nan
        elif new_velocity[1] == 0:
            new_angle = -0.5 * np.sign(new_velocity[0]) * np.pi
        else:
            if np.sign(new_velocity[1]) > 0:
                new_angle = np.arctan(-new_velocity[0] / new_velocity[1])
            else:
                new_angle = np.pi + np.arctan(
                    -new_velocity[0] / new_velocity[1])
        
        # Update the sprite's angle
        if (not np.isnan(new_angle) and
                np.abs(new_angle - sprite.angle) > _EPSILON):
            sprite.angle = new_angle
                
    def _update_sprite_in_maze(self, sprite):
        """Update a sprite velocity to abide by the maze."""
        position = sprite.position
        velocity = sprite.velocity
        
        if np.all(velocity == 0) or np.any(np.isnan(velocity)):
            return

        if self._max_speed is not None:
            velocity = np.clip(velocity, -self._max_speed, self._max_speed)
        
        if self._constant_speed is not None:
            velocity += np.sign(velocity)
            velocity *= self._constant_speed

        position, affordances = self._get_position_affordances(position)
        sprite.position = np.copy(position)
        new_velocity = self._get_new_velocity(position, velocity, affordances)
        
        # Update sprite angle and velocity
        self._update_sprite_angle(sprite, new_velocity)
        sprite.velocity = new_velocity

    def apply_physics(self, state, updates_per_env_step):
        """Move the sprites according to the physics."""
        if updates_per_env_step != 1:
            raise ValueError('Must have updates_per_env_step be 1 for maze.')
        for layer in self._avatar_layers:
            for sprite in state[layer]:
                self._update_sprite_in_maze(sprite)
