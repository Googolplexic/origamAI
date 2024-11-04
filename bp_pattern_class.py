import json
from typing import Dict, List, Union, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum


class CreaseType(Enum):
    NONE = 0
    MOUNTAIN = 1
    VALLEY = -1
    BORDER = 2  # Added for FOLD format compatibility


@dataclass
class Point:
    x: float  # Changed to float for FOLD compatibility
    y: float

    def __hash__(self):
        return hash((self.x, self.y))

    def to_list(self) -> List[float]:
        return [self.x, self.y]

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

    @classmethod
    def from_list(cls, coords: List[float]) -> "Point":
        return cls(coords[0], coords[1])


@dataclass
class Crease:
    start: Point
    end: Point
    type: CreaseType


class BoxPleatingPattern:
    def __init__(self, grid_size: int = None):
        """Initialize a box-pleating pattern with optional grid size."""
        self.grid_size = grid_size
        self.vertices: List[Point] = []
        self.creases: List[Crease] = []
        self.grid = None
        if grid_size is not None:
            self.grid = np.zeros((grid_size + 1, grid_size + 1), dtype=int)

    def __str__(self):
        """Return a string representation of the pattern."""

        pattern_data = {
            "grid_size": self.grid_size,
            "creases": [
                {
                    "start": {"x": crease.start.x, "y": crease.start.y},
                    "end": {"x": crease.end.x, "y": crease.end.y},
                    "type": crease.type.name,  # Assuming `type` is an Enum with a `.name`
                }
                for crease in self.creases
            ],
            "vertices": [{"x": vertex.x, "y": vertex.y} for vertex in self.vertices],
        }
        return json.dumps(pattern_data, indent=2)

    def _segments_intersect(
        self, p1: Point, p2: Point, p3: Point, p4: Point, epsilon: float = 1e-10
    ) -> bool:
        """
        Check if line segment p1p2 intersects with line segment p3p4.
        Handles all cases including overlapping lines.
        """
        # First check for complete overlap
        if (
            abs(p1.x - p3.x) < epsilon
            and abs(p1.y - p3.y) < epsilon
            and abs(p2.x - p4.x) < epsilon
            and abs(p2.y - p4.y) < epsilon
        ) or (
            abs(p1.x - p4.x) < epsilon
            and abs(p1.y - p4.y) < epsilon
            and abs(p2.x - p3.x) < epsilon
            and abs(p2.y - p3.y) < epsilon
        ):
            print("Complete overlap detected")
            return True

        # Check if segments share an endpoint (this is allowed)
        if (
            abs(p1.x - p3.x) < epsilon
            and abs(p1.y - p3.y) < epsilon
            or abs(p1.x - p4.x) < epsilon
            and abs(p1.y - p4.y) < epsilon
            or abs(p2.x - p3.x) < epsilon
            and abs(p2.y - p3.y) < epsilon
            or abs(p2.x - p4.x) < epsilon
            and abs(p2.y - p4.y) < epsilon
        ):
            return False

        def on_segment(p: Point, q: Point, r: Point) -> bool:
            """Check if point q lies on segment pr"""
            if (
                q.x <= max(p.x, r.x) + epsilon
                and q.x >= min(p.x, r.x) - epsilon
                and q.y <= max(p.y, r.y) + epsilon
                and q.y >= min(p.y, r.y) - epsilon
            ):
                # For box pleating, we also need to check if it's actually on the line
                # Calculate distance from point to line
                numerator = abs(
                    (r.y - p.y) * q.x - (r.x - p.x) * q.y + r.x * p.y - r.y * p.x
                )
                denominator = ((r.y - p.y) ** 2 + (r.x - p.x) ** 2) ** 0.5
                if denominator < epsilon:
                    return True
                distance = numerator / denominator
                return distance < epsilon
            return False

        def collinear(p1: Point, p2: Point, p3: Point) -> bool:
            """Check if three points are collinear using area of triangle"""
            area = abs(
                (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))
                / 2.0
            )
            return area < epsilon

        # Check if all points are collinear
        if collinear(p1, p2, p3) and collinear(p1, p2, p4):
            # Check for any overlap
            if (
                on_segment(p1, p3, p2)
                or on_segment(p1, p4, p2)
                or on_segment(p3, p1, p4)
                or on_segment(p3, p2, p4)
            ):
                print("Partial overlap detected")
                return True
            return False

        # Not collinear - check for regular intersection
        def direction(p1: Point, p2: Point, p3: Point) -> float:
            """Calculate direction of turn from p1->p2->p3"""
            return (p3.x - p1.x) * (p2.y - p1.y) - (p2.x - p1.x) * (p3.y - p1.y)

        d1 = direction(p3, p4, p1)
        d2 = direction(p3, p4, p2)
        d3 = direction(p1, p2, p3)
        d4 = direction(p1, p2, p4)

        # Check for intersection
        if ((d1 > epsilon and d2 < -epsilon) or (d1 < -epsilon and d2 > epsilon)) and (
            (d3 > epsilon and d4 < -epsilon) or (d3 < -epsilon and d4 > epsilon)
        ):
            print("Regular intersection detected")
            return True

        return False

    def add_crease(
        self, start: Point, end: Point, crease_type: CreaseType, force: bool = False
    ) -> bool:
        """Add a crease between two points if it follows box-pleating rules."""
        if not self._is_valid_crease(start, end):
            return False

        # Create temporary crease for checking
        new_crease = Crease(start, end, crease_type)

        # Check for intersections unless force is True
        if not force:
            for existing_crease in self.creases:
                # Check if creases are collinear
                v1 = Point(end.x - start.x, end.y - start.y)
                v2 = Point(
                    existing_crease.end.x - existing_crease.start.x,
                    existing_crease.end.y - existing_crease.start.y,
                )

                # Calculate cross product to check if vectors are parallel
                cross_product = v1.x * v2.y - v1.y * v2.x

                if abs(cross_product) < 1e-10:  # Vectors are parallel
                    # Check if the segments share any points or one is a subsegment of the other
                    def point_in_segment(
                        p: Point, seg_start: Point, seg_end: Point
                    ) -> bool:
                        """Check if point p lies on segment seg_start->seg_end"""
                        # Check if point is within bounding box of segment
                        if not (
                            min(seg_start.x, seg_end.x) - 1e-10
                            <= p.x
                            <= max(seg_start.x, seg_end.x) + 1e-10
                            and min(seg_start.y, seg_end.y) - 1e-10
                            <= p.y
                            <= max(seg_start.y, seg_end.y) + 1e-10
                        ):
                            return False

                        # Check if point lies on the line
                        if abs(seg_end.x - seg_start.x) < 1e-10:  # Vertical line
                            return abs(p.x - seg_start.x) < 1e-10
                        else:
                            # Check if point lies on the line using slope
                            slope = (seg_end.y - seg_start.y) / (
                                seg_end.x - seg_start.x
                            )
                            expected_y = seg_start.y + slope * (p.x - seg_start.x)
                            return abs(p.y - expected_y) < 1e-10

                    # Check if any endpoint of either segment lies on the other segment
                    if (
                        point_in_segment(
                            start, existing_crease.start, existing_crease.end
                        )
                        or point_in_segment(
                            end, existing_crease.start, existing_crease.end
                        )
                        or point_in_segment(existing_crease.start, start, end)
                        or point_in_segment(existing_crease.end, start, end)
                    ):
                        print(
                            f"Overlap detected: New crease ({start.x}, {start.y})->({end.x}, {end.y}) "
                            f"overlaps with existing crease ({existing_crease.start.x}, {existing_crease.start.y})->"
                            f"({existing_crease.end.x}, {existing_crease.end.y})"
                        )
                        return False
                else:
                    # Not parallel - check for regular intersection
                    if self._segments_intersect(
                        start, end, existing_crease.start, existing_crease.end
                    ):
                        print(
                            f"Intersection detected: New crease ({start.x}, {start.y})->({end.x}, {end.y}) "
                            f"intersects with existing crease ({existing_crease.start.x}, {existing_crease.start.y})->"
                            f"({existing_crease.end.x}, {existing_crease.end.y})"
                        )
                        return False

        # If we get here, the crease is valid or force is True - add it
        self.creases.append(new_crease)
        if start not in self.vertices:
            self.vertices.append(start)
        if end not in self.vertices:
            self.vertices.append(end)

        # Update grid
        self._update_grid(start, end, crease_type)
        return True

    def check_crease_intersections(self) -> List[Dict]:
        """
        Check for all intersecting creases in the pattern.
        """
        intersections = []

        # Check each pair of creases
        for i, crease1 in enumerate(self.creases):
            for j, crease2 in enumerate(self.creases[i + 1 :], i + 1):
                # Skip if creases share an endpoint
                if (
                    crease1.start == crease2.start
                    or crease1.start == crease2.end
                    or crease1.end == crease2.start
                    or crease1.end == crease2.end
                ):
                    continue

                # Check for intersection
                if self._segments_intersect(
                    crease1.start, crease1.end, crease2.start, crease2.end
                ):
                    # Calculate intersection point (simplified for grid-based system)
                    # For box-pleating, intersections can only occur at grid points
                    dx1 = crease1.end.x - crease1.start.x
                    dy1 = crease1.end.y - crease1.start.y
                    dx2 = crease2.end.x - crease2.start.x
                    dy2 = crease2.end.y - crease2.start.y

                    # For 45° and 90° lines in a grid, intersection must be at a grid point
                    t = 0.5  # This works for box-pleating grid intersections
                    x = crease1.start.x + t * dx1
                    y = crease1.start.y + t * dy1

                    intersection = {
                        "crease1": {
                            "start": {"x": crease1.start.x, "y": crease1.start.y},
                            "end": {"x": crease1.end.x, "y": crease1.end.y},
                            "type": crease1.type.name,
                        },
                        "crease2": {
                            "start": {"x": crease2.start.x, "y": crease2.start.y},
                            "end": {"x": crease2.end.x, "y": crease2.end.y},
                            "type": crease2.type.name,
                        },
                        "intersection_point": {"x": x, "y": y},
                    }
                    intersections.append(intersection)

        return intersections

    def is_valid_pattern(self) -> Tuple[bool, Dict]:
        """
        Check if the pattern is valid by verifying:
        1. Flat-foldability (Kawasaki and Maekawa theorems)
        2. No invalid crease intersections

        Returns:
            Tuple[bool, Dict]: (is_valid, detailed_report)
        """
        # Check flat-foldability
        is_foldable, violations = self.is_flat_foldable()

        # Check intersections
        intersections = self.check_crease_intersections()

        # Compile detailed report
        report = {
            "is_flat_foldable": is_foldable,
            "foldability_violations": violations,
            "has_intersections": len(intersections) > 0,
            "intersections": intersections,
        }

        # Pattern is valid if it's flat-foldable and has no intersections
        is_valid = is_foldable and len(intersections) == 0

        return is_valid, report

    def _is_valid_crease(self, start: Point, end: Point) -> bool:
        """Check if a proposed crease follows box-pleating rules."""
        # Check if points are within grid
        if not (
            0 <= start.x <= self.grid_size
            and 0 <= start.y <= self.grid_size
            and 0 <= end.x <= self.grid_size
            and 0 <= end.y <= self.grid_size
        ):
            return False

        # Check if it's a 45° or 90° angle
        dx = abs(end.x - start.x)
        dy = abs(end.y - start.y)

        # Must be either 45° (dx == dy) or 90° (dx == 0 or dy == 0)
        if not ((dx == dy) or (dx == 0) or (dy == 0)):
            return False

        return True

    def _update_grid(self, start: Point, end: Point, crease_type: CreaseType):
        """Update the grid with the new crease."""
        # For simplicity, just mark the endpoints
        self.grid[start.x, start.y] = 1
        self.grid[end.x, end.y] = 1

    def check_maekawa_theorem(self, vertex: Point) -> Tuple[bool, Dict]:
        """
        Check if Maekawa's theorem is satisfied at a vertex.
        Returns (is_satisfied, details_dict)

        Maekawa's theorem states that at any vertex, the difference between
        the number of mountain and valley creases must be 2.
        """
        mountain_count = 0
        valley_count = 0
        connected_creases = []

        # Edge vertices are always valid for our purposes
        if (
            vertex.x == 0
            or vertex.y == 0
            or vertex.x == self.grid_size
            or vertex.y == self.grid_size
        ):
            return True, {
                "is_edge_vertex": True,
                "mountain_count": 0,
                "valley_count": 0,
                "difference": 0,
            }

        # Count mountain and valley creases at this vertex
        for crease in self.creases:
            if crease.start == vertex or crease.end == vertex:
                connected_creases.append(crease)
                if crease.type == CreaseType.MOUNTAIN:
                    mountain_count += 1
                elif crease.type == CreaseType.VALLEY:
                    valley_count += 1

        difference = abs(mountain_count - valley_count)
        is_satisfied = difference == 2

        return is_satisfied, {
            "is_edge_vertex": False,
            "mountain_count": mountain_count,
            "valley_count": valley_count,
            "difference": difference,
            "connected_creases": len(connected_creases),
        }

    def _calculate_vector_angle(self, vec: np.ndarray) -> float:
        """Calculate the angle of a vector relative to positive x-axis."""
        return np.arctan2(vec[1], vec[0])

    def _sort_vectors_counterclockwise(
        self, vertex: Point, vertex_creases: List[Crease]
    ) -> List[Tuple[np.ndarray, Crease]]:
        """
        Sort vectors around a vertex in counterclockwise order.
        Returns list of (vector, crease) tuples to maintain crease association.
        """
        vectors_and_creases = []
        for crease in vertex_creases:
            if crease.start == vertex:
                vec = np.array([crease.end.x - vertex.x, crease.end.y - vertex.y])
            else:
                vec = np.array([crease.start.x - vertex.x, crease.start.y - vertex.y])
            vectors_and_creases.append((vec, crease))

        # Sort by angle with respect to positive x-axis
        angles = [self._calculate_vector_angle(v) for v, _ in vectors_and_creases]
        sorted_pairs = sorted(zip(angles, vectors_and_creases))
        return [vc for _, vc in sorted_pairs]

    def _calculate_angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Calculate the signed angle between two vectors, positive for counterclockwise.
        Returns angle in degrees.
        """
        # Calculate cross product to determine orientation
        cross_product = np.cross(v1, v2)

        # Calculate dot product and magnitudes
        dot_product = np.dot(v1, v2)
        magnitudes = np.linalg.norm(v1) * np.linalg.norm(v2)

        # Handle numerical precision
        cos_theta = np.clip(dot_product / magnitudes, -1.0, 1.0)
        angle = np.arccos(cos_theta)

        # Use cross product sign to determine orientation
        if cross_product < 0:
            angle = 2 * np.pi - angle

        return np.degrees(angle)

    def _calculate_angles(
        self, vertex: Point, vertex_creases: List[Crease]
    ) -> List[float]:
        """
        Calculate angles between creases at a vertex in cyclic order.
        Returns angles in degrees, sorted counterclockwise.
        """
        if len(vertex_creases) < 2:
            return []

        # Convert creases to vectors and sort them cyclically
        vectors_and_creases = []
        for crease in vertex_creases:
            # Always create vector pointing away from vertex
            if crease.start == vertex:
                vec = np.array([crease.end.x - vertex.x, crease.end.y - vertex.y])
            else:
                vec = np.array([crease.start.x - vertex.x, crease.start.y - vertex.y])
            vectors_and_creases.append((vec, crease))

        # Sort vectors by angle relative to positive x-axis
        angles_and_pairs = []
        for vec, crease in vectors_and_creases:
            angle = np.arctan2(vec[1], vec[0])
            # Normalize angle to [0, 2π)
            if angle < 0:
                angle += 2 * np.pi
            angles_and_pairs.append((angle, (vec, crease)))

        # Sort by angle
        sorted_pairs = sorted(angles_and_pairs, key=lambda x: x[0])
        sorted_vectors = [pair[1][0] for pair in sorted_pairs]

        # Calculate consecutive angles between vectors
        angles = []
        for i in range(len(sorted_vectors)):
            v1 = sorted_vectors[i]
            v2 = sorted_vectors[(i + 1) % len(sorted_vectors)]

            # Calculate angle between vectors using dot product
            dot_product = np.dot(v1, v2)
            norms_product = np.linalg.norm(v1) * np.linalg.norm(v2)

            # Handle numerical precision
            cos_theta = np.clip(dot_product / norms_product, -1.0, 1.0)
            angle = np.arccos(cos_theta)

            # Convert to degrees
            angle_degrees = np.degrees(angle)
            angles.append(angle_degrees)

        return angles

    def check_kawasaki_theorem(self, vertex: Point) -> Tuple[bool, Dict]:
        """
        Check if Kawasaki's theorem is satisfied at a vertex.
        Returns (is_satisfied, details_dict)

        Kawasaki's theorem states that at any vertex, the sum of alternate
        angles must be equal (sum of even-numbered angles = sum of odd-numbered angles).
        """
        if (
            vertex.x == 0
            or vertex.y == 0
            or vertex.x == self.grid_size
            or vertex.y == self.grid_size
        ):
            return True, {
                "is_edge_vertex": True,
                "angle_count": 0,
                "angle_difference": 0,
            }

        # Get all creases connected to this vertex
        vertex_creases = [
            c for c in self.creases if c.start == vertex or c.end == vertex
        ]

        if len(vertex_creases) < 4:  # Need at least 4 creases for internal vertex
            return False, {
                "is_edge_vertex": False,
                "angle_count": len(vertex_creases),
                "error": "Insufficient creases",
                "min_required": 4,
            }

        # Calculate angles between creases in cyclic order
        angles = self._calculate_angles(vertex, vertex_creases)

        # For Kawasaki's theorem, we need alternating sums
        sum_odd = sum(angles[1::2])  # Sum of odd-indexed angles
        sum_even = sum(angles[::2])  # Sum of even-indexed angles
        angle_difference = abs(sum_odd - sum_even)

        # Allow for small numerical imprecision (0.1 degrees)
        is_satisfied = angle_difference < 0.1

        return is_satisfied, {
            "is_edge_vertex": False,
            "angle_count": len(angles),
            "angles": angles,
            "sum_odd": sum_odd,
            "sum_even": sum_even,
            "angle_difference": angle_difference,
        }

    def is_flat_foldable(self) -> Tuple[bool, List[Dict]]:
        """
        Check if the entire pattern is flat-foldable and return detailed diagnostics.
        Returns (is_foldable, list_of_violations)
        """
        is_foldable = True
        violations = []

        for vertex in self.vertices:
            maekawa_valid, maekawa_details = self.check_maekawa_theorem(vertex)
            kawasaki_valid, kawasaki_details = self.check_kawasaki_theorem(vertex)

            if not (maekawa_valid and kawasaki_valid):
                is_foldable = False
                violation = {
                    "vertex": {"x": vertex.x, "y": vertex.y},
                    "maekawa_satisfied": "True" if maekawa_valid else "False",
                    "kawasaki_satisfied": "True" if kawasaki_valid else "False",
                    "maekawa_details": maekawa_details,
                    "kawasaki_details": kawasaki_details,
                }
                violations.append(violation)

                print(f"\nViolation at vertex ({vertex.x}, {vertex.y}):")
                if not maekawa_valid:
                    print(f"  Maekawa's theorem violated:")
                    print(f"    Mountain creases: {maekawa_details['mountain_count']}")
                    print(f"    Valley creases: {maekawa_details['valley_count']}")
                    print(
                        f"    Difference: {maekawa_details['difference']} (should be 2)"
                    )

                if not kawasaki_valid:
                    print(f"  Kawasaki's theorem violated:")
                    if "error" in kawasaki_details:
                        print(f"    {kawasaki_details['error']}")
                        print(
                            f"    Found {kawasaki_details['angle_count']} creases, need {kawasaki_details['min_required']}"
                        )
                    else:
                        print(
                            f"    Sum of odd angles: {kawasaki_details['sum_odd']:.2f}°"
                        )
                        print(
                            f"    Sum of even angles: {kawasaki_details['sum_even']:.2f}°"
                        )
                        print(
                            f"    Difference: {kawasaki_details['angle_difference']:.2f}°"
                        )
                        print(
                            f"    Angles: {[f'{a:.1f}°' for a in kawasaki_details['angles']]}"
                        )

        return is_foldable, violations

    def _are_edges_parallel(
        self, edge1: Crease, edge2: Crease, epsilon: float = 1e-6
    ) -> bool:
        """
        Check if two edges/creases are parallel within a given epsilon.

        Args:
            edge1: First crease
            edge2: Second crease
            epsilon: Tolerance for floating point comparisons

        Returns:
            bool: True if edges are parallel, False otherwise
        """
        # Calculate direction vectors
        vec1 = np.array([edge1.end.x - edge1.start.x, edge1.end.y - edge1.start.y])
        vec2 = np.array([edge2.end.x - edge2.start.x, edge2.end.y - edge2.start.y])

        # Normalize vectors
        vec1_norm = np.linalg.norm(vec1)
        vec2_norm = np.linalg.norm(vec2)

        if vec1_norm < epsilon or vec2_norm < epsilon:
            return False

        vec1 = vec1 / vec1_norm
        vec2 = vec2 / vec2_norm

        # Check if vectors are parallel (or anti-parallel)
        cross_product = abs(np.cross(vec1, vec2))
        return cross_product < epsilon

    def _is_corner_vertex(self, vertex: Point) -> bool:
        """
        Check if a vertex is at the corner of the pattern.

        Args:
            vertex: Point to check

        Returns:
            bool: True if vertex is at a corner, False otherwise
        """
        # In a box-pleating pattern, corners are at the grid boundaries
        return (
            (vertex.x == 0 and vertex.y == 0)
            or (vertex.x == 0 and vertex.y == self.grid_size)
            or (vertex.x == self.grid_size and vertex.y == 0)
            or (vertex.x == self.grid_size and vertex.y == self.grid_size)
        )

    def _is_boundary_vertex(self, vertex: Point) -> bool:
        """
        Check if a vertex is on the boundary of the pattern.

        Args:
            vertex: Point to check

        Returns:
            bool: True if vertex is on boundary, False otherwise
        """
        return (
            vertex.x == 0
            or vertex.x == self.grid_size
            or vertex.y == 0
            or vertex.y == self.grid_size
        )

    def _get_connected_creases(self, vertex: Point) -> List[Crease]:
        """
        Get all creases connected to a vertex.

        Args:
            vertex: Point to find connected creases for

        Returns:
            List[Crease]: List of creases connected to the vertex
        """
        return [
            crease
            for crease in self.creases
            if crease.start == vertex or crease.end == vertex
        ]

    def _should_remove_vertex(
        self, vertex: Point, connected_creases: List[Crease]
    ) -> bool:
        """
        Determine if a vertex should be removed based on box-pleating rules.

        Args:
            vertex: Point to check
            connected_creases: List of creases connected to the vertex

        Returns:
            bool: True if vertex should be removed, False otherwise
        """
        # Only consider vertices with exactly two creases
        if len(connected_creases) != 2:
            return False

        # Don't remove corner vertices
        if self._is_corner_vertex(vertex):
            return False

        crease1, crease2 = connected_creases

        # Creases must be of same type (mountain, valley, or border)
        if crease1.type != crease2.type:
            return False

        # Creases must be parallel
        if not self._are_edges_parallel(crease1, crease2):
            return False

        return True

    def _merge_creases(
        self, vertex: Point, creases_to_merge: List[Crease]
    ) -> Optional[Crease]:
        """
        Merge two creases that meet at a vertex into a single crease.

        Args:
            vertex: Common vertex between creases
            creases_to_merge: List of two creases to merge

        Returns:
            Optional[Crease]: New merged crease, or None if merge is invalid
        """
        if len(creases_to_merge) != 2:
            return None

        crease1, crease2 = creases_to_merge

        # Find the endpoints that aren't the common vertex
        if crease1.start == vertex:
            start = crease1.end
        else:
            start = crease1.start

        if crease2.start == vertex:
            end = crease2.end
        else:
            end = crease2.start

        # Create new merged crease
        return Crease(start, end, crease1.type)

    def remove_redundant_vertices(self):
        """
        Remove redundant vertices in the pattern that don't affect its structure.
        This includes vertices where two parallel creases of the same type meet
        and the vertex isn't at a corner.
        """
        vertices_to_remove = []
        creases_to_remove = []
        creases_to_add = []

        # First pass: identify vertices and creases to modify
        for vertex in self.vertices:
            connected_creases = self._get_connected_creases(vertex)

            if self._should_remove_vertex(vertex, connected_creases):
                vertices_to_remove.append(vertex)
                creases_to_remove.extend(connected_creases)

                # Create merged crease
                new_crease = self._merge_creases(vertex, connected_creases)
                if new_crease:
                    creases_to_add.append(new_crease)

        # Second pass: perform modifications
        # Remove old creases first to avoid conflicts
        self.creases = [c for c in self.creases if c not in creases_to_remove]

        # Add merged creases
        self.creases.extend(creases_to_add)

        # Remove vertices
        self.vertices = [v for v in self.vertices if v not in vertices_to_remove]

        # Debug information
        for vertex in vertices_to_remove:
            print(f"Removed redundant vertex at ({vertex.x}, {vertex.y})")
        print(
            f"Removed {len(creases_to_remove)} creases and added {len(creases_to_add)} merged creases"
        )
