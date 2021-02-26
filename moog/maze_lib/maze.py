"""Maze class.

This file contains the Maze class, which is a class to mediate between a binary
maze array and sprites, including methods to infer a maze from sprites, convert
a maze to sprites, and sample points or connected blobs in the maze.
"""

import numpy as np
from moog import sprite

# Small error tolerance for checking whether wall vertex values are multiples of
# a grid size
_EPSILON = 1e-4

# Sanity check to prevent infinite while looping. No maze should have grid size
# larger than this
_MAX_MAZE_SIZE = 100


class Maze():
    """Maze class."""

    def __init__(self, maze):
        """Constructor.

        Args:
            maze: Square numpy array with binary values (may be a boolean array,
                or an int/float array with 0/1 values). The 1's are the walls of
                the maze.
        """
        self.maze = maze
        self.maze_size = maze.shape[0]
        self.grid_side = 1. / self.maze_size
        self.half_grid_side = 0.5 * self.grid_side
        self.side_vertices = np.linspace(
            self.half_grid_side, 1. - self.half_grid_side, self.maze_size)

    @classmethod
    def from_state(cls, state, maze_layer='walls'):
        """Get a maze object from environment state.

        This method works by first inferring the maze size by looking for the
        smallest integer N such that 1/N divides the vertex values of the
        sprites in the maze values. Then it checks whether each of the
        centerpoints of an NxN grid are contained in a maze wall sprite to
        compute the maze matrix.
        
        Args:
            state: OrderedDict of sprites. State of the environment.
            maze_layer: String. Layer in the environment containing the maze
                wall sprites.
        """
        wall_vertices = np.array([s.vertices for s in state[maze_layer]])
        
        # Find the smallest maze size N such that all wall vertices are a
        # multiple of 1/N
        maze_size = 1
        valid_maze_size = False
        while not valid_maze_size:
            if maze_size > _MAX_MAZE_SIZE:
                raise ValueError(
                    'Cannot find a maze grid size. Your maze sprites are '
                    'invalid.')
            rounded_vertices = np.round(wall_vertices * maze_size) / maze_size
            if np.allclose(rounded_vertices, wall_vertices, atol=_EPSILON):
                valid_maze_size = True
            else:
                maze_size += 1
        
        # Get a matrix of grid square centerpoints
        half_grid_side = 1. / (2 * maze_size)
        edge_centers = np.linspace(
            half_grid_side, 1 - half_grid_side, maze_size)
        grid_centers = np.stack(np.meshgrid(edge_centers, edge_centers), axis=2)
        flat_grid_centers = np.reshape(grid_centers, (maze_size * maze_size, 2))

        # Now check which grid square centerpoints are inside walls
        maze = np.zeros(maze_size * maze_size, dtype=bool)
        for s in state[maze_layer]:
            contained = s.contains_points(flat_grid_centers)
            maze = np.logical_or(maze, contained)

        maze = np.reshape(maze, (maze_size, maze_size)).astype(int)
        return cls(maze)

    def to_sprites(self, **color):
        """Turn a maze matrix into a list of sprites.
        
        Each sprite is a single square for one brick in the maze walls.

        Args:
            color: Dict. May contain keys {'c0', 'c1', 'c2'} for the wall sprite
                color factors.

        Returns:
            sprites: List of maze wall sprites.
        """
        maze_size = self.maze_size
        vertex_linspace = np.linspace(0., 1., maze_size + 1)
        sprites = []
        for x in range(maze_size):
            for y in range(maze_size):
                if self.maze[y, x]:
                    shape = np.array([
                        [vertex_linspace[x], vertex_linspace[y]],
                        [vertex_linspace[x], vertex_linspace[y + 1]],
                        [vertex_linspace[x + 1], vertex_linspace[y + 1]],
                        [vertex_linspace[x + 1], vertex_linspace[y]],
                    ])
                    new_sprite = sprite.Sprite(x=0., y=0., shape=shape, **color)
                    sprites.append(new_sprite)
        return sprites

    def open_vertex(self, i, j):
        """Returns whether the (i, j) cell is open, i.e. not a maze wall."""
        if i < 0 or j < 0 or i >= self.maze_size or j >= self.maze_size:
            return False
        else:
            return not self.maze[j, i]

    def valid_directions(self, i, j):
        """Computes the open neighbords of the (i, j) cell."""
        valid_directions = np.array([
            [self.open_vertex(k, j) for k in [i - 1, i + 1]],
            [self.open_vertex(i, k) for k in [j - 1, j + 1]],
        ])
        return valid_directions

    def sample_random_position(self, off_intersection=True):
        """Sample random open position on the edges of the maze."""
        # First find edges
        neg_maze = 1 - self.maze
        v_edges = np.logical_and(neg_maze[1:], neg_maze[:-1])
        h_edges = np.logical_and(neg_maze[:, 1:], neg_maze[:, :-1])
        v_edges = np.stack(np.nonzero(v_edges)[::-1]).T
        h_edges = np.stack(np.nonzero(h_edges)[::-1]).T
        num_h_edges = len(h_edges)
        num_v_edges = len(v_edges)
        num_edges = num_h_edges + num_v_edges
        if np.random.rand() < float(num_h_edges) / num_edges:
            # Pick a horizontal edge
            edge = h_edges[np.random.choice(num_h_edges)]
            position = edge
            if off_intersection:
                position = edge + np.random.rand() * np.array([1., 0.])
        else:
            # Pick a vertical edge
            edge = v_edges[np.random.choice(num_v_edges)]
            position = edge
            if off_intersection:
                position = edge + np.random.rand() * np.array([0., 1.])
        position = self.half_grid_side + position * self.grid_side
        return position

    def to_background_grid(self, line_thickness=0.01, **color):
        """Get static grid of background lines as list of sprites.

        These lines are in the channels of the maze and when rendered can
        provide a nice visual effect.
        
        Args:
            line_thickness: Float. How thick the grid lines should be.
            color: Dict. May contain keys {'c0', 'c1', 'c2'} for the background
                line sprite color factors.

        Returns:
            grid_sprites: List of sprites, the grid lines.
        """
        grid_sprites = []
        def _add_sprite(min_x, max_x, min_y, max_y):
            shape = np.array([
                [min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]
            ])
            grid_sprites.append(sprite.Sprite(x=0., y=0., shape=shape, **color))
        
        for x in self.side_vertices:
            min_x = x - 0.5 * line_thickness
            max_x = x + 0.5 * line_thickness
            _add_sprite(min_x, max_x, 0., 1.)

        for y in self.side_vertices:
            min_y = y - 0.5 * line_thickness
            max_y = y + 0.5 * line_thickness
            _add_sprite(0., 1., min_y, max_y)

        return grid_sprites

    def sample_open_point(self):
        """Sample an open point in the maze.

        Returns:
            Tuple of integers (i, j), coordinates of open point.
        """
        if np.sum(1 - self.maze) == 0:
            raise ValueError('Maze has no open point.')
        candidates = np.argwhere(self.maze == 0)
        point = candidates[np.random.randint(len(candidates))]
        return tuple(point)

    def sample_distinct_open_points(self, num_points):
        """Sample distinct open points in the maze.

        Args:
            num_points: Int. Number of distinct open points to sample.

        Returns:
            Tuple of integers (i, j), coordinates of open point.
        """
        if np.sum(1 - self.maze) < num_points:
            raise ValueError('Maze has no open point.')
        candidates = np.argwhere(self.maze == 0)
        inds = np.random.choice(len(candidates), size=num_points, replace=False)
        points = [tuple(candidates[i]) for i in inds]
        return points
    
    def get_neighbors(self, i, j):
        """Get list of neighbors of point (i, j)."""
        neighbors = []
        if i > 0 and not self.maze[i - 1, j]:
            neighbors.append((i - 1, j))
        if i < self.maze_size - 1 and not self.maze[i + 1, j]:
            neighbors.append((i + 1, j))
        if j > 0 and not self.maze[i, j - 1]:
            neighbors.append((i, j - 1))
        if j < self.maze_size - 1 and not self.maze[i, j + 1]:
            neighbors.append((i, j + 1))
        return neighbors
    
    def get_neighbor_dict(self):
        """Get dictionary of neighbors of all points.
        
        Returns:
            neighbor_dict: Keys are int tuples (i, j) and values are lists of
                int tuples for all open neighbors of (i, j).
        """
        neighbor_dict = {
            (i, j): self.get_neighbors(i, j)
            for i in range(self.maze_size)
            for j in range(self.maze_size)
        }
        return neighbor_dict

    def add_wall(self, x_range, y_range):
        """Add a rectangular wall to the maze.

        Args:
            x_range: start and end point of the wall in the x-coordinate
            y_range: start and end point of the wall in the y-coordinate
        """
        self.maze[x_range[0] : x_range[1] + 1, y_range[0] : y_range[1] + 1] = 1

    def add_outer_walls(self):
        """Make the maze borders walls.
        
        Warning: This could distrupt properties of the maze. For example, if the
        maze originally had no dead ends, this could introduce dead ends.
        """
        self.maze[0, :] = 1
        self.maze[-1, :] = 1
        self.maze[:, 0] = 1
        self.maze[:, -1] = 1
