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


class FOLDConverter:
    """Handles conversion between BoxPleatingPattern and FOLD format"""

    @staticmethod
    def assignment_to_crease_type(assignment: str) -> CreaseType:
        """Convert FOLD assignment to CreaseType"""
        if assignment == "M":
            return CreaseType.MOUNTAIN
        elif assignment == "V":
            return CreaseType.VALLEY
        elif assignment == "B":
            return CreaseType.BORDER
        else:
            return CreaseType.NONE

    @staticmethod
    def crease_type_to_assignment(crease_type: CreaseType) -> str:
        """Convert CreaseType to FOLD assignment"""
        if crease_type == CreaseType.MOUNTAIN:
            return "M"
        elif crease_type == CreaseType.VALLEY:
            return "V"
        elif crease_type == CreaseType.BORDER:
            return "B"
        else:
            return "F"  # Flat/unassigned

    @staticmethod
    def crease_type_to_fold_angle(crease_type: CreaseType) -> float:
        """Convert CreaseType to fold angle"""
        if crease_type == CreaseType.MOUNTAIN:
            return -180.0
        elif crease_type == CreaseType.VALLEY:
            return 180.0
        else:
            return 0.0


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
                    "type": crease.type.name  # Assuming `type` is an Enum with a `.name`
                }
                for crease in self.creases
            ],
            "vertices": [
                {"x": vertex.x, "y": vertex.y}
                for vertex in self.vertices
            ]
        }
        return json.dumps(pattern_data, indent=2)

    def add_crease(self, start: Point, end: Point, crease_type: CreaseType) -> bool:
        """Add a crease between two points if it follows box-pleating rules."""
        if not self._is_valid_crease(start, end):
            return False

        # Add the crease
        self.creases.append(Crease(start, end, crease_type))
        if start not in self.vertices:
            self.vertices.append(start)
        if end not in self.vertices:
            self.vertices.append(end)

        # Update grid
        self._update_grid(start, end, crease_type)
        return True

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
        if (vertex.x == 0 or vertex.y == 0 or 
            vertex.x == self.grid_size or 
            vertex.y == self.grid_size):
            return True, {
                "is_edge_vertex": True,
                "mountain_count": 0,
                "valley_count": 0,
                "difference": 0
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
            "connected_creases": len(connected_creases)
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


    def _calculate_angles(self, vertex: Point, vertex_creases: List[Crease]) -> List[float]:
        """
        Calculate angles between creases at a vertex.
        Returns angles in degrees, sorted counterclockwise.
        """
        if len(vertex_creases) < 2:
            return []

        # Sort vectors counterclockwise and maintain crease association
        vec_crease_pairs = self._sort_vectors_counterclockwise(vertex, vertex_creases)
        vectors = [v for v, _ in vec_crease_pairs]

        # Calculate angles between consecutive vectors
        angles = []
        for i in range(len(vectors)):
            v1 = vectors[i]
            v2 = vectors[(i + 1) % len(vectors)]
            angle = self._calculate_angle_between_vectors(v1, v2)
            angles.append(angle)

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
            return True, {"is_edge_vertex": True, "angle_count": 0, "angle_difference": 0}

        # Get all creases connected to this vertex
        vertex_creases = [c for c in self.creases if c.start == vertex or c.end == vertex]

        if len(vertex_creases) < 4:  # Need at least 4 creases for internal vertex
            return False, {
                "is_edge_vertex": False,
                "angle_count": len(vertex_creases),
                "error": "Insufficient creases",
                "min_required": 4,
            }

        # Calculate angles between creases
        angles = self._calculate_angles(vertex, vertex_creases)
        angles.sort()  # Sort for consistent comparison

        # Sum alternate angles
        sum_odd = sum(angles[1::2])
        sum_even = sum(angles[::2])
        angle_difference = abs(sum_odd - sum_even)

        # Allow for small floating point imprecision
        is_satisfied = angle_difference < 0.001

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
                    print(f"    Difference: {maekawa_details['difference']} (should be 2)")

                if not kawasaki_valid:
                    print(f"  Kawasaki's theorem violated:")
                    if "error" in kawasaki_details:
                        print(f"    {kawasaki_details['error']}")
                        print(
                            f"    Found {kawasaki_details['angle_count']} creases, need {kawasaki_details['min_required']}"
                        )
                    else:
                        print(f"    Sum of odd angles: {kawasaki_details['sum_odd']:.2f}°")
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
    
    def _are_edges_parallel(self, edge1: Crease, edge2: Crease, epsilon: float = 1e-6) -> bool:
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
        return (vertex.x == 0 and vertex.y == 0) or \
            (vertex.x == 0 and vertex.y == self.grid_size) or \
            (vertex.x == self.grid_size and vertex.y == 0) or \
            (vertex.x == self.grid_size and vertex.y == self.grid_size)

    def _is_boundary_vertex(self, vertex: Point) -> bool:
        """
        Check if a vertex is on the boundary of the pattern.
        
        Args:
            vertex: Point to check
            
        Returns:
            bool: True if vertex is on boundary, False otherwise
        """
        return vertex.x == 0 or vertex.x == self.grid_size or \
            vertex.y == 0 or vertex.y == self.grid_size

    def _get_connected_creases(self, vertex: Point) -> List[Crease]:
        """
        Get all creases connected to a vertex.
        
        Args:
            vertex: Point to find connected creases for
            
        Returns:
            List[Crease]: List of creases connected to the vertex
        """
        return [crease for crease in self.creases 
                if crease.start == vertex or crease.end == vertex]

    def _should_remove_vertex(self, vertex: Point, connected_creases: List[Crease]) -> bool:
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

    def _merge_creases(self, vertex: Point, creases_to_merge: List[Crease]) -> Optional[Crease]:
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
        print(f"Removed {len(creases_to_remove)} creases and added {len(creases_to_add)} merged creases")
