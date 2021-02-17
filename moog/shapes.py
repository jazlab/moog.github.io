"""Shapes and shape-fetching functions for common use across tasks."""

import numpy as np
from moog import sprite
from spriteworld import shapes

# A selection of simple shapes. Elements in SHAPES can be looked up from their
# string keys in sprite.Sprite, i.e. you can give a string key as the `shape`
# argument to sprite.Sprite and it will fetch the vertices if that key is in
# this dictionary.
SHAPES = {
    'triangle': shapes.polygon(num_sides=3, theta_0=np.pi/2),
    'square': shapes.polygon(num_sides=4, theta_0=np.pi/4),
    'pentagon': shapes.polygon(num_sides=5, theta_0=np.pi/2),
    'hexagon': shapes.polygon(num_sides=6),
    'octagon': shapes.polygon(num_sides=8),
    'circle': shapes.polygon(num_sides=30),
    'star_4': shapes.star(num_sides=4, theta_0=np.pi/4),
    'star_5': shapes.star(num_sides=5, theta_0=np.pi + np.pi/10),
    'star_6': shapes.star(num_sides=6),
    'spoke_4': shapes.spokes(num_sides=4, theta_0=np.pi/4),
    'spoke_5': shapes.spokes(num_sides=5, theta_0=np.pi + np.pi/10),
    'spoke_6': shapes.spokes(num_sides=6),
}


def border_walls(visible_thickness=0.05,
                 total_thickness=0.5,
                 c0=0,
                 c1=0,
                 c2=0,
                 opacity=255):
    """Get four sprites forming a border around the [0, 1] x [0, 1] frame.

    This can be used to (i) create the visual effect of a border at the edges of
    the screen, and/or (ii) create walls around the border that can be used to
    contain sprites inside the interior of the screen.

    Args:
        visible_thickness: Float. How thick the borders within the frame should
            be.
        total_thickness: Float. How thick the border wall is in total. Depending
            on visible_thickness, much of this may lie outside of the frame. As
            long as total_thickness is greater than visible_thickness, it is not
            important. However, if visible_thickness is very small then it can
            be good to have total_thickness non-negligibly greater than zero,
            otherwise the wall sprites are extremely narrow and collisions can
            be a little unstable since their vertices and centers of mass are
            nearly collinear.
        c0: Scalar. First coordinate of color of wall sprites.
        c1: Scalar. Second coordinate of color of wall sprites.
        c2: Scalar. Third coordinate of color of wall sprites.
        opacity: Integer in [0, 255]. Opacity of wall sprites.

    Returns:
        walls: List of four sprites, the walls.
    """
    boundary_wall_shape_0 = np.array([
        [0., visible_thickness],
        [1., visible_thickness],
        [1., visible_thickness - total_thickness],
        [0., visible_thickness - total_thickness],
    ])
    distance_across_frame = 1 + total_thickness - 2 * visible_thickness
    wall_shapes = [
        boundary_wall_shape_0,
        boundary_wall_shape_0 + np.array([[0., distance_across_frame]]),
        np.flip(boundary_wall_shape_0, axis=1),
        np.flip(boundary_wall_shape_0, axis=1) + np.array(
            [[distance_across_frame, 0.]]),
    ]
    sprite_factors = dict(x=0., y=0., c0=c0, c1=c1, c2=c2, opacity=opacity)
    walls = [
        sprite.Sprite(shape=wall_shape, **sprite_factors)
        for wall_shape in wall_shapes
    ]
    return walls


def grid_lines(grid_x=0.4,
               grid_y=0.4,
               line_thickness=0.01,
               buffer_border=0.,
               c0=0,
               c1=0,
               c2=0,
               opacity=255):
    """Get grid of lines.

    Returns a list of thin rectangular sprites forming grid lines.
    
    This is sometimes used to put a grid in the background, particularly when
    using a first-person renderer for which this grid tells the player how the
    agent is moving.

    Args:
        grid_x: Float. Width of each grid cell.
        grid_y: Float. Height of each grid cell.
        line_thickness: Float. How thick the grid lines should be.
        buffer_border: Float. How far around the frame in every direction to
            create the grid. This is useful for first-person rendering, when the
            field of view sometimes extends outside [0, 1] x [0, 1].
        c0: Scalar. First coordinate of color of background grid sprites.
        c1: Scalar. Second coordinate of color of background grid sprites.
        c2: Scalar. Third coordinate of color of background grid sprites.
        opacity: Integer in [0, 255]. Opacity of background grid sprites.

    Returns:
        grid_lines: List of sprites, the grid lines.
    """
    half_num_lines_across = int(np.floor((0.5 + buffer_border) / grid_x))
    half_num_lines_up = int(np.floor((0.5 + buffer_border) / grid_y))

    x_vertices = np.linspace(
        start=0.5 - half_num_lines_across * grid_x,
        stop=0.5 + half_num_lines_across * grid_x,
        num=1 + 2 * half_num_lines_across,
    )

    y_vertices = np.linspace(
        start=0.5 - half_num_lines_up * grid_y,
        stop=0.5 + half_num_lines_up * grid_y,
        num=1 + 2 * half_num_lines_up,
    )

    sprite_factors = dict(x=0., y=0., c0=c0, c1=c1, c2=c2, opacity=opacity)
    grid_sprites = []
    def _add_sprite(min_x, max_x, min_y, max_y):
        shape = np.array([
            [min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]
        ])
        grid_sprites.append(sprite.Sprite(shape=shape, **sprite_factors))

    for x in x_vertices:
        min_x = x - 0.5 * line_thickness
        max_x = x + 0.5 * line_thickness
        min_y = -1 * buffer_border
        max_y = 1. + buffer_border
        _add_sprite(min_x, max_x, min_y, max_y)

    for y in y_vertices:
        min_x = -1 * buffer_border
        max_x = 1. + buffer_border
        min_y = y - 0.5 * line_thickness
        max_y = y + 0.5 * line_thickness
        _add_sprite(min_x, max_x, min_y, max_y)

    return grid_sprites


def circle_vertices(radius, num_sides=50):
    """Get vertices for a circle, centered about the origin.
    
    Args:
        radius: Scalar. Radius of the circle.
        num_sides: Int. Number of sides in the circle. The circle is really just
            a many-sided polygon with this many sides
        
    Returns:
        circle: Numpy array of shape [num_sides, 2] containing the vertices of
            the circle.
    """
    min_theta = 2 * np.pi / num_sides
    thetas = np.linspace(min_theta, 2 * np.pi, num_sides)
    circle = np.stack([np.sin(thetas), np.cos(thetas)], axis=1)
    circle *= radius
    return circle


def annulus_vertices(inner_radius, outer_radius, num_sides=50):
    """Get vertices for an annulus, centered about the origin.
    
    Args:
        inner_radius: Float. Radius of inner circle
        outer_radius: Float. Radius of outer circle.
        num_sides: Int. Number of sides in each circle. Each circle is really a
            many-sided polygon.

    Returns:
        annulus: Numpy array of shape [num_sides, 2] containing the vertices of
            the annulus.
    """
    inner_circle = circle_vertices(inner_radius, num_sides=num_sides)
    inner_circle = np.concatenate((inner_circle, [inner_circle[0]]), axis=0)
    outer_circle = circle_vertices(outer_radius, num_sides=num_sides)
    outer_circle = np.concatenate((outer_circle, [outer_circle[0]]), axis=0)
    annulus = np.concatenate((inner_circle, outer_circle[::-1]), axis=0)
    return annulus
