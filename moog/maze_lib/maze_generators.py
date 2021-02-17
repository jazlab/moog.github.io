"""This file contains functions to randomly generate mazes.

The main function in this file is generate_random_maze_matrix(), which generates
a maze with no dead ends and no open squares.
"""

import numpy as np

# Maximum iteration through a while loop
_MAX_ITERS = int(1e5)


def _get_neighbors(size, point):
    """Get indices of point's neighbors in square matrix of size `size`.

    Unless point (i, j) is on the boundary of the size x size square, this will
    be a list of 4 elements.

    Args:
        size: Int.
        point: Tuple of ints (i, j). Must satisfy 0 <= i, j < size.

    Returns:
        neighbors: List of tuples. Length 2 (if point is a corner), 3 (if point
            is on an edge), or 4 (if point is in the interior).
    """
    i, j = point

    neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
    _valid_neighbor = lambda neighbor: all(0 <= x < size for x in neighbor)
    neighbors = list(filter(_valid_neighbor, neighbors))
    
    return neighbors


def _get_containing_blocks(size, point):
    """Get 2x2 blocks containing point in open maze of size `size`.

    Unless point is on the boundary of the size x size square, there will be 4
    containing 2x2 blocks.

    Args:
        size: Int.
        point: Tuple of ints (i, j). Must satisfy 0 <= i, j < size.

    Returns:
        block_inds: List of tuples. If (k, l) is in block_inds, then point
            (i, j) is in {(k, l), (k + 1, l), (k, l + 1), (k + 1, l + 1)}.
    """
    i, j = point
    block_inds = []
    if i > 0:
        if j > 0:
            block_inds.append((i - 1, j - 1))
        if j < size - 1:
            block_inds.append((i - 1, j))
    if i < size - 1:
        if j > 0:
            block_inds.append((i, j - 1))
        if j < size - 1:
            block_inds.append((i, j))
            
    return block_inds


def _remove_dead_ends(maze):
    """Iteratively remove the dead ends in the maze.

    This is done in place, and by the time this function returs there will be no
    open points with fewer than 2 open neighbors, i.e. no dead ends in the maze.

    Args:
        maze: N x N binary matrix.
    """
    def _fill_maze():
        # Fill in dead ends, return True if the maze has no dead ends, otherwise
        # False.
        size = maze.shape[0]
        for i in range(size):
            for j in range(size):
                if maze[i, j]:  # Not an open point
                    continue
                num_open_neighbors = np.sum(
                    [1 - maze[n[0], n[1]]
                    for n in _get_neighbors(size, (i, j))])
                if num_open_neighbors < 2:
                    maze[i, j] = 1
                    return False
        return True

    valid_maze = False
    while not valid_maze:
        valid_maze = _fill_maze()
    

def generate_random_maze_matrix(size, ambient_size=None):
    """Generate a random maze matrix.

    The maze matrix generated has no open points (e.g. no four open cells
    sharing a vertex), no dead ends (e.g. each open point has at least two open
    neighbors), and is one connected component.

    The way this generator works is it starts will a single open cell (all other
    cells are walls). Then it iteratively adds closed neighbors, as long as
    opening them doesn't open up a block. Once there are no more such neighbors,
    it iteratively fills in all dead ends. The result is the final maze matrix
    (unless it is all walls, in which case the function recurses to try again).

    Args:
        size: Int. Size (height and width) of the maze.
        ambient_size: Size of the final maze matrix. This can be larger than
            `size` to add some visible wall border around the maze. If None, no
            wall border around the maze is produced.
    """
    maze = np.ones((size, size))

    # Start from a random point and recursively open points
    closed_neighbors = []  # Closed points that are neighbors of open points
    
    def _open_point(point):
        # Open a point and add its neighbors to closed_neighbors
        for p in _get_neighbors(size, point):
            if maze[p[0], p[1]] and p not in closed_neighbors:
                closed_neighbors.append(p)
        maze[point[0], point[1]] = 0

    def _find_and_open_new_point():
        # Find a closed neighbor that can be opened without creating an open
        # block, open it, and return True. If no such point exists, return
        # False.
        np.random.shuffle(closed_neighbors)
        for new_point in closed_neighbors:
            if not maze[new_point[0], new_point[1]]:
                continue
            will_make_open_block = any([
                np.sum(maze[i: i + 2, j: j + 2]) <= 1
                for i, j in _get_containing_blocks(size, new_point)
            ])
            if not will_make_open_block:
                _open_point(new_point)
                return True
        return False

    # Seed the maze and iteratively open points
    _open_point(tuple(np.random.randint(0, size, size=(2,))))
    points_to_add = True
    while points_to_add:
        points_to_add = _find_and_open_new_point()

    # Remove dead ends
    _remove_dead_ends(maze)
    
    # If maze has no open points, recurse to generate a new one
    if np.sum(1 - maze) == 0:
        return generate_random_maze_matrix(size, ambient_size=ambient_size)

    # Add wall border if necessary
    if ambient_size is not None and ambient_size > size:
        maze_with_border = np.ones((ambient_size, ambient_size))
        start_index = (ambient_size - size) // 2
        maze_with_border[start_index: start_index + size,
                         start_index: start_index + size] = maze
        maze = maze_with_border

    return maze


def _generate_open_blob(maze, num_points):
    """Try to generate an open connected blob of points in the maze.
    
    Args:
        maze: Instance of .maze.Maze.
        num_points: Int. Number of connected points to have in the blob.

    Returns:
        blob_matrix: False or binary matrix of same size as maze. Ones
            correspond to points in the connected open blob. If False, then
            could not generate a valid blob.
    """
    neighbor_dict = maze.get_neighbor_dict()

    # Seed the blob with a starting point
    blob = [maze.sample_open_point()]
    
    def _get_candidate_new_blob_point():
        # New potential new blob point from neighbors of existing blob point
        candidate_root = blob[np.random.randint(len(blob))]  #pylint: disable=invalid-sequence-index
        neighbors = neighbor_dict[candidate_root]
        candidate = neighbors[np.random.randint(len(neighbors))]
        return candidate
    
    def _add_point():
        # Add a new point to the blob, returning True/False depending on whether
        # this was successful.
        valid_candidate = False
        count = 0
        while not valid_candidate:
            count += 1
            candidate = _get_candidate_new_blob_point()
            if not candidate or candidate in blob:
                continue
            else:
                valid_candidate = True

            if count > _MAX_ITERS:
                return False
        blob.append(candidate)
        return True

    # Add num_points points to the blob if possible, else return False.
    for _ in range(num_points - 1):
        able_to_add_point = _add_point()
        if not able_to_add_point:
            return False

    # Convert the list of blob points to a matrix
    blob_matrix = np.zeros_like(maze.maze)
    for (i, j) in blob:
        blob_matrix[i, j] = 1

    return blob_matrix


def get_connected_open_blob(maze, num_points):
    """Generate an open connected blob of `num_points` points in the maze.
    
    Args:
        maze: Instance of .maze.Maze.
        num_points: Int. Number of connected points to have in the blob.

    Returns:
        blob_matrix: Binary matrix of same size as maze. Ones correspond to
            points in the connected open blob.
    """
    valid_blob = False
    count = 0
    while not valid_blob:
        count += 1
        if count > _MAX_ITERS:
            raise ValueError('Could not generate an open connected blob.')
        
        blob_matrix = _generate_open_blob(maze, num_points)
        if not isinstance(blob_matrix, bool):
            valid_blob = True
    
    return blob_matrix
