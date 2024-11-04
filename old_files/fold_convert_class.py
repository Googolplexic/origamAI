from typing import Dict, List, Union, Tuple, Set
import json
from old_files.bp_pattern_class import BoxPleatingPattern, Point, CreaseType, Crease
import math
import numpy as np


class FoldConverter:

    def _compute_minimum_spacing(self, vertices: List[List[float]]) -> float:
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

        return min_spacing if min_spacing != float("inf") else 1.0

    def _compute_optimal_grid_size(
        self, vertices: List[List[float]], edges: List[List[int]]
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
        min_spacing = self._compute_minimum_spacing(vertices)

        # Calculate minimum required grid size to preserve smallest details
        min_required_size = math.ceil(max_range / min_spacing)

        # No padding needed - the grid size should be exactly what's required
        grid_size = min_required_size

        # Ensure grid size stays within reasonable bounds
        grid_size = max(2, min(grid_size, 50))

        return grid_size

    def _compute_scale_factors(
        self, coords: List[List[float]]
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
            scale_factor = 400 / max_dimension if max_dimension != 0 else 1.0

        return scale_factor, min_x, min_y, center_x, center_y

    def _snap_to_grid(self, value: float, grid_size: int) -> int:
        """
        Snap a value to the nearest grid point while preserving relative positions.
        """
        grid_value = value * grid_size
        return int(round(grid_value))

    def _preserve_parallel_lines(
        self, points: List[Point], grid_size: int
    ) -> List[Point]:
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

    def from_fold(self, fold_data: Dict) -> BoxPleatingPattern:
        """Create BoxPleatingPattern from FOLD format with minimal grid size."""
        vertices = fold_data["vertices_coords"]
        for i in range(len(vertices)):
            vertices[i] = [coord * 12 for coord in vertices[i]]
            vertices[i] = [round(coord * 10) / 10 for coord in vertices[i]]
        edges = fold_data["edges_vertices"]

        grid_size = self._compute_optimal_grid_size(vertices, edges)
        pattern = BoxPleatingPattern(grid_size)

        vertex_points = []
        if vertices:
            xs = [x for x, _ in vertices]
            ys = [y for _, y in vertices]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            x_range = max_x - min_x
            y_range = max_y - min_y

            for x, y in vertices:
                norm_x = (x - min_x) / (x_range if x_range != 0 else 1)
                norm_y = (y - min_y) / (y_range if y_range != 0 else 1)

                grid_x = round(norm_x * (grid_size if grid_size > 1 else 1))
                grid_y = round(norm_y * (grid_size if grid_size > 1 else 1))

                vertex_points.append(Point(grid_x, grid_y))

        for (v1_idx, v2_idx), assignment in zip(
            edges, fold_data.get("edges_assignment", ["U"] * len(edges))
        ):
            crease_type = CreaseType.NONE
            if assignment == "M":
                crease_type = CreaseType.MOUNTAIN
            elif assignment == "V":
                crease_type = CreaseType.VALLEY

            start = vertex_points[v1_idx]
            end = vertex_points[v2_idx]
            pattern.add_crease(start, end, crease_type)

        return pattern

    def _compute_faces_vertices(
        self, vertices_coords: List[List[float]], edges_vertices: List[List[int]]
    ) -> List[List[int]]:
        """
        Compute faces from edges following FOLD spec approach.
        Returns list of vertex indices for each face in counterclockwise order.
        """
        # First create vertices_vertices (adjacency list)
        num_vertices = len(vertices_coords)
        vertices_vertices = [[] for _ in range(num_vertices)]
        for v1, v2 in edges_vertices:
            vertices_vertices[v1].append(v2)
            vertices_vertices[v2].append(v1)

        # Sort vertices around each vertex counterclockwise
        for v, neighbors in enumerate(vertices_vertices):
            if not neighbors:
                continue
            # Calculate angles for sorting
            angles = []
            for n in neighbors:
                dx = vertices_coords[n][0] - vertices_coords[v][0]
                dy = vertices_coords[n][1] - vertices_coords[v][1]
                angle = math.atan2(dy, dx)
                angles.append((angle, n))
            # Sort neighbors counterclockwise
            sorted_pairs = sorted(angles)
            vertices_vertices[v] = [n for _, n in sorted_pairs]

        # Build next mapping from sorted neighbors
        next_map = {}
        for v, neighbors in enumerate(vertices_vertices):
            for i, n in enumerate(neighbors):
                prev = neighbors[(i - 1) % len(neighbors)]
                next_map[(v, n)] = prev

        # Find faces
        faces = []
        for start_edge in edges_vertices:
            # Try both directions of each edge
            for v1, v2 in [
                (start_edge[0], start_edge[1]),
                (start_edge[1], start_edge[0]),
            ]:
                face = [v1, v2]
                while True:
                    if len(face) > len(edges_vertices):
                        break

                    curr = face[-1]
                    prev = face[-2]
                    next_v = next_map.get((curr, prev))

                    if next_v is None:
                        break

                    if next_v == face[0]:
                        # Check if this forms a valid CCW face
                        area = 0
                        for i in range(len(face)):
                            j = (i + 1) % len(face)
                            vi = vertices_coords[face[i]]
                            vj = vertices_coords[face[j]]
                            area += vi[0] * vj[1] - vj[0] * vi[1]

                        if area > 0 and len(face) >= 3:
                            # Check if this is a new face (not just a cyclic rotation)
                            face_set = frozenset(face)
                            if not any(
                                frozenset(existing) == face_set for existing in faces
                            ):
                                faces.append(face[:])
                        break

                    if next_v in face[:-1]:
                        break

                    face.append(next_v)

        return faces

    def to_fold(self, pattern: BoxPleatingPattern) -> Dict:
        """Export BoxPleatingPattern to FOLD format."""
        vertices_coords = []
        vertex_map = {}
        for vertex in pattern.vertices:
            vertex_map[vertex] = len(vertices_coords)
            x = (vertex.x - pattern.grid_size / 2) * (400 / pattern.grid_size)
            y = (vertex.y - pattern.grid_size / 2) * (400 / pattern.grid_size)
            vertices_coords.append([x, y])

        edges_vertices = []
        edges_assignment = []
        edges_foldAngle = []

        for crease in pattern.creases:
            v1 = vertex_map[crease.start]
            v2 = vertex_map[crease.end]
            edges_vertices.append([v1, v2])

            if crease.type == CreaseType.MOUNTAIN:
                edges_assignment.append("M")
                edges_foldAngle.append(-180)
            elif crease.type == CreaseType.VALLEY:
                edges_assignment.append("V")
                edges_foldAngle.append(180)
            else:
                edges_assignment.append("B")
                edges_foldAngle.append(0)

        # Compute faces
        faces_vertices = self._compute_faces_vertices(vertices_coords, edges_vertices)

        # Build vertices_vertices from edges
        vertices_vertices = [[] for _ in range(len(vertices_coords))]
        for v1, v2 in edges_vertices:
            vertices_vertices[v1].append(v2)
            vertices_vertices[v2].append(v1)

        for neighbors in vertices_vertices:
            neighbors.sort()

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
            "faces_vertices": faces_vertices,
        }

    def save_fold(self, pattern: BoxPleatingPattern, filename: str):
        """Save BoxPleatingPattern to FOLD file."""

        fold_data = self.to_fold(pattern)
        with open(filename, "w") as f:
            json.dump(fold_data, f, indent=2)

    def load_fold(self, filename: str) -> BoxPleatingPattern:
        """Load BoxPleatingPattern from FOLD file."""
        with open(filename, "r") as f:
            fold_data = json.load(f)
        return self.from_fold(fold_data)
