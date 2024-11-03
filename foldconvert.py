from typing import Dict, List, Union, Tuple, Set
import json
from app import BoxPleatingPattern, Point, CreaseType, Crease
import math
import numpy as np


class FoldConverter:

    @staticmethod
    @staticmethod
    def _compute_minimum_spacing(vertices: List[List[float]]) -> float:
        """
        Compute the minimum non-zero spacing between vertices in the pattern.
        """
        if len(vertices) < 2:
            return 1.0

        min_spacing = float("inf")

        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                dx = abs(vertices[i][0] - vertices[j][0])
                dy = abs(vertices[i][1] - vertices[j][1])

                # Consider non-zero distances
                if dx > 0:
                    min_spacing = min(min_spacing, dx)
                if dy > 0:
                    min_spacing = min(min_spacing, dy)
        # If no non-zero spacing found, return 1.0 as default
        return min_spacing if min_spacing != float("inf") else 1.0

    @staticmethod
    def _compute_optimal_grid_size(vertices: List[List[float]]) -> int:
        """
        Compute the optimal grid size based on pattern geometry.
        """
        if not vertices:
            return 4  # Default size for empty patterns

        # Find coordinate ranges
        xs = [x for x, _ in vertices]
        ys = [y for _, y in vertices]
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        max_range = max(x_range, y_range)

        # Find minimum spacing between vertices
        min_spacing = FoldConverter._compute_minimum_spacing(vertices)

        # Calculate minimum required grid size to preserve smallest details
        min_required_size = math.ceil(max_range / min_spacing)

        # Ensure grid size stays within reasonable bounds
        grid_size = max(4, min(min_required_size, 96))

        return grid_size

    def _compute_scale_factors(
        coords: List[List[float]],
    ) -> Tuple[float, float, float, float, float]:
        """
        Compute scale factors and offsets for coordinate conversion.
        Returns (scale_factor, min_x, min_y, center_x, center_y)
        """
        if not coords:
            return 1.0, 0.0, 0.0, 0.0, 0.0

        # Extract x and y coordinates
        xs = [x for x, _ in coords]
        ys = [y for _, y in coords]

        # Calculate bounds
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Calculate centers
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        # Calculate scale factor based on the larger dimension
        width = max_x - min_x
        height = max_y - min_y

        # Prevent division by zero
        if width == 0 and height == 0:
            scale_factor = 1.0
        else:
            # Use the larger dimension to determine scale
            max_dimension = max(width, height)
            if max_dimension == 0:
                scale_factor = 1.0
            else:
                # Target a reasonable size (e.g., 400 units total width/height)
                scale_factor = 400 / max_dimension

        return scale_factor, min_x, min_y, center_x, center_y

    @staticmethod
    def _snap_to_grid(value: float, grid_size: int) -> int:
        """
        Snap a value to the nearest grid point while preserving relative positions.
        """
        # Scale to grid coordinates
        grid_value = value * grid_size
        # Round to nearest integer while preserving order
        return int(round(grid_value))

    @staticmethod
    def _preserve_parallel_lines(points: List[Point], grid_size: int) -> List[Point]:
        """
        Adjust points to preserve parallel lines and 45-degree angles.
        """
        # Group points by x and y coordinates
        points_by_x = {}
        points_by_y = {}
        for p in points:
            if p.x not in points_by_x:
                points_by_x[p.x] = []
            if p.y not in points_by_y:
                points_by_y[p.y] = []
            points_by_x[p.x].append(p)
            points_by_y[p.y].append(p)

        # Adjust points that should be aligned
        adjusted_points = []
        for p in points:
            new_x = p.x
            new_y = p.y

            # Check if point should be aligned vertically
            if len(points_by_x[p.x]) > 1:
                # Snap to nearest grid line
                new_x = round(p.x)

            # Check if point should be aligned horizontally
            if len(points_by_y[p.y]) > 1:
                # Snap to nearest grid line
                new_y = round(p.y)

            adjusted_points.append(Point(new_x, new_y))

        return adjusted_points

    @staticmethod
    def to_fold(pattern: BoxPleatingPattern) -> Dict:
        """
        Export BoxPleatingPattern to FOLD format.
        Uses standard 400x400 coordinate system centered at origin (-200 to +200).
        """
        vertices_coords = []
        vertex_map = {}  # Maps Point to vertex index

        # Convert grid coordinates to FOLD coordinates (-200 to +200)
        for vertex in pattern.vertices:
            vertex_map[vertex] = len(vertices_coords)
            # Convert from grid position to -200 to +200 range
            x = (vertex.x - pattern.grid_size / 2) * (400 / pattern.grid_size)
            y = (vertex.y - pattern.grid_size / 2) * (400 / pattern.grid_size)
            vertices_coords.append([x, y])

        # Create edges information
        edges_vertices = []
        edges_assignment = []
        edges_foldAngle = []

        # Initialize vertices_vertices (adjacency list)
        vertices_vertices = [[] for _ in range(len(vertices_coords))]

        for crease in pattern.creases:
            v1 = vertex_map[crease.start]
            v2 = vertex_map[crease.end]
            edges_vertices.append([v1, v2])

            # Update adjacency lists
            vertices_vertices[v1].append(v2)
            vertices_vertices[v2].append(v1)

            # Convert CreaseType to FOLD assignment and fold angle
            if crease.type == CreaseType.MOUNTAIN:
                edges_assignment.append("M")
                edges_foldAngle.append(-180)
            elif crease.type == CreaseType.VALLEY:
                edges_assignment.append("V")
                edges_foldAngle.append(180)
            else:
                edges_assignment.append("B")
                edges_foldAngle.append(0)

        # Sort the adjacency lists
        for adj_list in vertices_vertices:
            adj_list.sort()

        return {
            "file_spec": 1.1,
            "file_creator": "BoxPleatingPattern Converter",
            "file_classes": ["creasePattern"],
            "frame_classes": ["creasePattern"],
            "vertices_coords": vertices_coords,
            "edges_vertices": edges_vertices,
            "edges_assignment": edges_assignment,
            "edges_foldAngle": edges_foldAngle,
            "vertices_vertices": vertices_vertices,
        }

    @staticmethod
    def from_fold(fold_data: Dict) -> BoxPleatingPattern:
        """Create BoxPleatingPattern from FOLD format with optimized grid size."""
        vertices = fold_data["vertices_coords"]

        # Compute scaling factors
        scale_factor, min_x, min_y, center_x, center_y = (
            FoldConverter._compute_scale_factors(vertices)
        )

        # Compute optimal grid size based on pattern geometry
        grid_size = FoldConverter._compute_optimal_grid_size(vertices)

        pattern = BoxPleatingPattern(grid_size)

        # Convert vertices to Points
        vertex_points = []
        for x, y in vertices:
            # Center the coordinates
            centered_x = x - center_x
            centered_y = y - center_y

            # Scale to grid
            grid_x = FoldConverter._snap_to_grid(
                (centered_x / (400 / 2)) + 0.5, grid_size
            )
            grid_y = FoldConverter._snap_to_grid(
                (centered_y / (400 / 2)) + 0.5, grid_size
            )

            # Ensure points stay within grid bounds
            grid_x = max(0, min(grid_size, grid_x))
            grid_y = max(0, min(grid_size, grid_y))

            vertex_points.append(Point(grid_x, grid_y))

        # Preserve parallel lines and 45-degree angles
        vertex_points = FoldConverter._preserve_parallel_lines(vertex_points, grid_size)

        # Add creases
        edges = fold_data["edges_vertices"]
        assignments = fold_data.get(
            "edges_assignment", ["U"] * len(edges)
        )  # Default to unassigned

        for (v1_idx, v2_idx), assignment in zip(edges, assignments):
            if assignment == "M":
                crease_type = CreaseType.MOUNTAIN
            elif assignment == "V":
                crease_type = CreaseType.VALLEY
            else:
                crease_type = CreaseType.NONE

            start = vertex_points[v1_idx]
            end = vertex_points[v2_idx]
            pattern.add_crease(start, end, crease_type)

        return pattern

    @staticmethod
    def save_fold(pattern: BoxPleatingPattern, filename: str):
        """Save BoxPleatingPattern to FOLD file."""
        fold_data = FoldConverter.to_fold(pattern)
        with open(filename, "w") as f:
            json.dump(fold_data, f, indent=2)

    @staticmethod
    def load_fold(filename: str) -> BoxPleatingPattern:
        """Load BoxPleatingPattern from FOLD file."""
        with open(filename, "r") as f:
            fold_data = json.load(f)
        return FoldConverter.from_fold(fold_data)


def test_fold_converter():
    # Test loading the example FOLD file
    # Convert to BoxPleatingPattern
    with open("input.fold", "r") as f:
        example_fold = json.load(f)
    pattern = FoldConverter.from_fold(example_fold)
    print(pattern.grid_size, "grid size")

    print(f"Is flat-foldable: {pattern.is_flat_foldable()}")


if __name__ == "__main__":
    test_fold_converter()
