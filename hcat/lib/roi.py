import math
from typing import List


def get_squares(vertices: List[List[float]], square_size: int, overlap: float):
    """
    Returns a list of square verticies of squares which are at least partially in
    a polygon.

    :param vertices: Verticies of a Polygon [[x0, y0], ...]
    :param square_size: Size of square
    :param overlap: percentage overlap of the boxes
    :return: list of box coordinates
    """

    assert overlap < 1, f'square overlap must be less than 1, not {overlap}'
    assert square_size > 1, f'square size must be greater than 1, not {square_size}'

    square_size=float(square_size)

    # Calculate the bounding box of the polygon
    min_x = min(x for x, y in vertices)
    max_x = max(x for x, y in vertices)
    min_y = min(y for x, y in vertices)
    max_y = max(y for x, y in vertices)

    # Calculate the amount of overlap in each direction
    overlap_x = square_size * overlap
    overlap_y = square_size * overlap

    # Calculate the number of squares needed in each direction
    num_squares_x = math.ceil((max_x - min_x + overlap_x) / (square_size - overlap_x))
    num_squares_y = math.ceil((max_y - min_y + overlap_y) / (square_size - overlap_y))

    # Generate the list of squares

    num_point_on_edge = 5
    fraction = 1 / 5

    squares = []
    for i in range(num_squares_x):
        for j in range(num_squares_y):
            x = min_x + i * (square_size - overlap_x)
            y = min_y + j * (square_size - overlap_y)
            square_vertices = [
                (x, y),
                (x + square_size, y),
                (x + square_size, y + square_size),
                (x, y + square_size),
            ]

            #left x0->x0, y0->y1
            left = [(x, y+(square_size*fraction* i)) for i in range(num_point_on_edge)]

            #top
            top = [(x+(square_size*fraction * i), y+square_size) for i in range(num_point_on_edge)]

            #right
            right = [(x+square_size, y+(square_size*fraction* i)) for i in range(num_point_on_edge)]

            #bottom
            bottom = [(x+(square_size*fraction * i), y) for i in range(num_point_on_edge)]

            square_vertices.extend(left)
            square_vertices.extend(top)
            square_vertices.extend(right)
            square_vertices.extend(bottom)

            # Check if the square has at least one corner inside the polygon
            if num_squares_x > 1 and num_squares_y > 1:
                if any(is_point_inside_polygon(v, vertices) for v in square_vertices):
                    squares.append(square_vertices)
            else:
                squares.append(square_vertices)

    squares = [(s[0][0], s[0][1], square_size, square_size) for s in squares]
    return squares # [x0, y0, w, h]


def is_point_inside_polygon(point: List[float],
                            vertices: List[List[float]]) -> bool:
    """
    Calculates if a point is inside a polygon.

    :param point: [x0, y0]
    :param vertices: List[[x0, y0], ...]

    :return: True if a point is inside, false otherwise
    """
    # Implementation of the Ray Casting Algorithm
    # Source: https://stackoverflow.com/a/21337692
    x, y = point
    inside = False
    for i in range(len(vertices)):
        j = (i + 1) % len(vertices)
        if ((vertices[i][1] > y) != (vertices[j][1] > y) and
                (x < (vertices[j][0] - vertices[i][0]) * (y - vertices[i][1]) / (vertices[j][1] - vertices[i][1]) +
                 vertices[i][0])):
            inside = not inside
    return inside