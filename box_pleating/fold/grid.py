import math
from typing import List
from ..models import Point


def compute_optimal_grid_size(
    vertices: List[List[float]], edges: List[List[int]]
) -> int:
    """
    Compute the optimal grid size based on pattern geometry.
    """
    if not vertices:
        return 1  # Empty pattern

    if len(vertices) == 1:
        return 1  # Single point

    if len(vertices) == 2 and len(edges) == 1:
        return 2  # Single line (any orientation)

    # Find coordinate ranges
    xs = [x for x, _ in vertices]
    ys = [y for _, y in vertices]
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    max_range = max(x_range, y_range)

    # Find minimum spacing between vertices
    from .geometry import compute_minimum_spacing

    min_spacing = compute_minimum_spacing(vertices)

    # Calculate minimum required grid size to preserve smallest details
    min_required_size = math.ceil(max_range / min_spacing)

    # No padding needed - the grid size should be exactly what's required
    grid_size = min_required_size

    # Ensure grid size stays within reasonable bounds
    grid_size = max(2, min(grid_size, 50))

    return grid_size


def snap_to_grid(value: float, grid_size: int) -> int:
    """
    Snap a value to the nearest grid point while preserving relative positions.
    """
    grid_value = value * grid_size
    return int(round(grid_value))


def preserve_parallel_lines(points: List[Point], grid_size: int) -> List[Point]:
    """
    Adjust points to preserve parallel lines and 45-degree angles.
    """
    points_by_x = {}
    points_by_y = {}
    for p in points:
        if p.x not in points_by_x:
            points_by_x[p.x] = []
        if p.y not in points_by_y:
            points_by_y[p.y] = []
        points_by_x[p.x].append(p)
        points_by_y[p.y].append(p)

    adjusted_points = []
    for p in points:
        new_x = p.x
        new_y = p.y

        if len(points_by_x[p.x]) > 1:
            new_x = round(p.x)
        if len(points_by_y[p.y]) > 1:
            new_y = round(p.y)

        adjusted_points.append(Point(new_x, new_y))

    return adjusted_points
