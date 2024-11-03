from bp_pattern_class import BoxPleatingPattern, Point, CreaseType
from typing import List, Tuple
from fold_convert_class import FoldConverter

class BoxPleatingGenerator:
    def __init__(self, grid_size: int = 8):
        self.grid_size = grid_size
        self.pattern = BoxPleatingPattern(grid_size)

    def add_mountain_valley_pair(
        self, start1: Point, end1: Point, start2: Point, end2: Point
    ) -> bool:
        """
        Add a mountain-valley crease pair that maintains local flat-foldability.
        Returns True if successful, False if the addition would violate constraints.
        """
        # First check if the creases are valid box-pleating lines
        if not (
            self.pattern._is_valid_crease(start1, end1)
            and self.pattern._is_valid_crease(start2, end2)
        ):
            return False

        # Temporarily add creases
        self.pattern.add_crease(start1, end1, CreaseType.MOUNTAIN)
        self.pattern.add_crease(start2, end2, CreaseType.VALLEY)

        # Check flat-foldability
        is_foldable = self.pattern.is_flat_foldable()

        if not is_foldable[0]:
            # Remove creases if not flat-foldable
            self.pattern.creases = self.pattern.creases[:-2]
            self.pattern.vertices = list(
                set(v for c in self.pattern.creases for v in [c.start, c.end])
            )
            return False

        return True

    def generate_base_pattern(self) -> None:
        """
        Generate a basic flat-foldable pattern as a starting point.
        Creates a simple box-pleating base with mountain and valley creases.
        """
        # Create central square
        center = self.grid_size // 2
        points = [
            Point(center - 1, center - 1),
            Point(center + 1, center - 1),
            Point(center + 1, center + 1),
            Point(center - 1, center + 1),
        ]

        # Add mountain creases for the square
        for i in range(4):
            self.pattern.add_crease(points[i], points[(i + 1) % 4], CreaseType.MOUNTAIN)

        # Add valley creases from corners
        for i in range(4):
            self.pattern.add_crease(
                points[i],
                Point(
                    points[i].x + (1 if i in [0, 3] else -1),
                    points[i].y + (1 if i in [0, 1] else -1),
                ),
                CreaseType.VALLEY,
            )

    def extend_pattern(self, iterations: int = 3) -> None:
        """
        Extend the pattern by adding more mountain-valley pairs.
        Uses a guided random approach to maintain flat-foldability.
        """
        for _ in range(iterations):
            # Find potential extension points
            vertices = set(self.pattern.vertices)

            # Try adding new mountain-valley pairs
            for v1 in vertices:
                for v2 in vertices:
                    if v1 == v2:
                        continue

                    # Find valid box-pleating directions from these vertices
                    directions = self._get_valid_directions(v1, v2)

                    for dir1, dir2 in directions:
                        end1 = Point(v1.x + dir1[0], v1.y + dir1[1])
                        end2 = Point(v2.x + dir2[0], v2.y + dir2[1])

                        if self.add_mountain_valley_pair(v1, end1, v2, end2):
                            break  # Successfully added a pair

    def _get_valid_directions(
        self, p1: Point, p2: Point
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Get valid box-pleating directions from two points.
        Returns pairs of directions for mountain and valley creases.
        """
        # Box-pleating allows 45° and 90° angles
        basic_dirs = [
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
            (0, -1),
            (1, -1),
        ]

        valid_pairs = []
        for dir1 in basic_dirs:
            for dir2 in basic_dirs:
                # Check if directions maintain box-pleating rules
                if (abs(dir1[0]) == abs(dir1[1]) or dir1[0] * dir1[1] == 0) and (
                    abs(dir2[0]) == abs(dir2[1]) or dir2[0] * dir2[1] == 0
                ):
                    valid_pairs.append((dir1, dir2))

        return valid_pairs


def generate_flat_foldable_pattern(
    grid_size: int = 8, complexity: int = 3
) -> BoxPleatingPattern:
    """
    Generate a complete flat-foldable box-pleating pattern.
    """
    generator = BoxPleatingGenerator(grid_size)
    generator.generate_base_pattern()
    generator.extend_pattern(complexity)
    return generator.pattern


# Create a new pattern
pattern = generate_flat_foldable_pattern(grid_size=8, complexity=3)

# Convert to FOLD format
converter = FoldConverter()
fold_data = converter.to_fold(pattern)

# Save to file
converter.save_fold(pattern, "output.fold")
