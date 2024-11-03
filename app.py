import json
from typing import Dict, List, Union, Tuple
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
    def from_list(cls, coords: List[float]) -> 'Point':
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


        creases_str = ', '.join([f"({c.start.x}, {c.start.y}) -> ({c.end.x}, {c.end.y}) [{c.type.name}]" for c in self.creases])
        vertices_str = ', '.join([f"({v.x}, {v.y})" for v in self.vertices])
        return (f"BoxPleatingPattern(grid_size={self.grid_size}, creases=[{creases_str}], vertices=[{vertices_str}])")

    
    @classmethod
    def from_fold(cls, fold_data: Dict) -> 'BoxPleatingPattern':
        """Create a BoxPleatingPattern from FOLD format data"""
        pattern = cls()
        
        # Convert vertices
        pattern.vertices = [Point.from_list(coords) 
            for coords in fold_data['vertices_coords']]
        
        # Convert edges/creases
        for i, (v1_idx, v2_idx) in enumerate(fold_data['edges_vertices']):
            assignment = fold_data['edges_assignment'][i]
            crease_type = FOLDConverter.assignment_to_crease_type(assignment)
            pattern.creases.append(Crease(
                pattern.vertices[v1_idx],
                pattern.vertices[v2_idx],
                crease_type
            ))
        
        return pattern

    def to_fold(self) -> Dict:
        """Convert the pattern to FOLD format"""
        # Create vertices coords list and build vertex mapping
        vertex_to_idx = {}
        vertices_coords = []
        for i, vertex in enumerate(self.vertices):
            vertex_to_idx[vertex] = i
            vertices_coords.append(vertex.to_list())
        
        # Create edges data
        edges_vertices = []
        edges_assignment = []
        edges_foldAngle = []
        
        for crease in self.creases:
            v1_idx = vertex_to_idx[crease.start]
            v2_idx = vertex_to_idx[crease.end]
            edges_vertices.append([v1_idx, v2_idx])
            
            assignment = FOLDConverter.crease_type_to_assignment(crease.type)
            edges_assignment.append(assignment)
            
            fold_angle = FOLDConverter.crease_type_to_fold_angle(crease.type)
            edges_foldAngle.append(fold_angle)
        
        # Create FOLD format dictionary
        fold_data = {
            "file_spec": 1.1,
            "file_creator": "box_pleating_ai",
            "vertices_coords": vertices_coords,
            "edges_vertices": edges_vertices,
            "edges_assignment": edges_assignment,
            "edges_foldAngle": edges_foldAngle
        }
        
        return fold_data

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
        if not (0 <= start.x <= self.grid_size and 0 <= start.y <= self.grid_size and
                0 <= end.x <= self.grid_size and 0 <= end.y <= self.grid_size):
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

    def check_maekawa_theorem(self, vertex: Point) -> bool:
        """Check if Maekawa's theorem is satisfied at a vertex."""
        mountain_count = 0
        valley_count = 0
        if vertex.x == 0 or vertex.y == 0 or vertex.x == self.grid_size or vertex.y == self.grid_size:
            # Edge vertex
            return True

        for crease in self.creases:
            if crease.start == 0 or crease.end == 0 or crease.start == self.grid_size or crease.end == self.grid_size:
                continue
            if crease.start == vertex or crease.end == vertex:
                if crease.type == CreaseType.MOUNTAIN:
                    mountain_count += 1
                elif crease.type == CreaseType.VALLEY:
                    valley_count += 1

        return abs(mountain_count - valley_count) == 2

    def check_kawasaki_theorem(self, vertex: Point) -> bool:
        """Check if Kawasaki's theorem is satisfied at a vertex."""

        if vertex.x == 0 or vertex.y == 0 or vertex.x == self.grid_size or vertex.y == self.grid_size:
            # Edge vertex
            return True

        # Get all creases connected to this vertex
        vertex_creases = [c for c in self.creases if c.start == vertex or c.end == vertex]

        if len(vertex_creases) < 4:  # Need at least 4 creases for a vertex
            return False

        # Calculate angles between creases
        angles = self._calculate_angles(vertex, vertex_creases)

        # Sort angles
        angles.sort()

        # Sum alternate angles
        sum_odd = sum(angles[1::2])
        sum_even = sum(angles[::2])

        # Allow for some floating point imprecision
        return abs(sum_odd - sum_even) < 0.001

    def _calculate_angles(self, vertex: Point, vertex_creases: List[Crease]) -> List[float]:
        """Calculate the angles between creases at a vertex."""
        angles = []
        for i, crease1 in enumerate(vertex_creases):
            # Get vectors from vertex to crease endpoints
            vec1 = self._get_vector(vertex, crease1)
            # Get next crease
            crease2 = vertex_creases[(i + 1) % len(vertex_creases)]
            vec2 = self._get_vector(vertex, crease2)
            # Calculate angle between vectors
            angle = np.arccos(np.dot(vec1, vec2) / 
                            (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
            angles.append(np.degrees(angle))
        return angles

    def _get_vector(self, vertex: Point, crease: Crease) -> np.ndarray:
        """Get the vector from the vertex to the crease endpoint."""
        if crease.start == vertex:
            return np.array([crease.end.x - vertex.x, crease.end.y - vertex.y])
        else:
            return np.array([crease.start.x - vertex.x, crease.start.y - vertex.y])

    def is_flat_foldable(self) -> bool:
        """Check if the entire pattern is flat-foldable."""
        # Check each vertex satisfies both Maekawa and Kawasaki theorems
        for vertex in self.vertices:
            if not (self.check_maekawa_theorem(vertex) and
                self.check_kawasaki_theorem(vertex)):
                return False
        return True

def read_fold_file(filename: str) -> BoxPleatingPattern:
    """Read a FOLD file and return a BoxPleatingPattern"""
    with open(filename, 'r') as f:
        fold_data = json.load(f)
    return BoxPleatingPattern.from_fold(fold_data)

def write_fold_file(pattern: BoxPleatingPattern, filename: str):
    """Write a BoxPleatingPattern to a FOLD file"""
    fold_data = pattern.to_fold()
    with open(filename, 'w') as f:
        json.dump(fold_data, f, indent=4)

# Helper function to create basic molecules
def create_temp_crease_pattern(pattern: BoxPleatingPattern, center: Point):
    """Create a placeholder pattern"""
    # Points for a basic waterbomb base
    top = Point(center.x, center.y - 5)
    bottom = Point(center.x, center.y + 5)
    left = Point(center.x - 5, center.y)
    right = Point(center.x + 5, center.y)

    # Add creases
    pattern.add_crease(top, center, CreaseType.MOUNTAIN)
    pattern.add_crease(bottom, center, CreaseType.MOUNTAIN)
    pattern.add_crease(left, center, CreaseType.VALLEY)
    pattern.add_crease(right, center, CreaseType.MOUNTAIN)

def main():
    # Example usage
    pattern = BoxPleatingPattern(10)  # 10x10 grid
    pattern = read_fold_file("input.fold")
    # Create a simple waterbomb base
    center = Point(5, 5)
    # create_temp_crease_pattern(pattern, center)

    # Check if it's flat-foldable
    print(f"Pattern is flat-foldable: {pattern.is_flat_foldable()}")

if __name__ == "__main__":
    main()