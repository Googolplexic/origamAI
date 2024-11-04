import random
from typing import List, Tuple, Set
from bp_pattern_class import BoxPleatingPattern, Point, CreaseType
from fold_convert_class import FoldConverter


class BoxPleatingGenerator:
    def __init__(self, grid_size: int = 4):
        """Initialize the pattern generator with a given grid size."""
        self.grid_size = grid_size

    def generate_base_grid(self) -> BoxPleatingPattern:
        """Generate a basic grid pattern that serves as a starting point."""
        pattern = BoxPleatingPattern(self.grid_size)

        # Add horizontal and vertical grid lines
        for i in range(1, self.grid_size):
            # Horizontal lines
            start = Point(0, i)
            end = Point(self.grid_size, i)
            pattern.add_crease(start, end, CreaseType.MOUNTAIN)

            # Vertical lines
            start = Point(i, 0)
            end = Point(i, self.grid_size)
            pattern.add_crease(start, end, CreaseType.VALLEY)
            pattern.add_crease(start, end, CreaseType.MOUNTAIN)

        # Add border creases
        topLeft = Point(0, 0)
        topRight = Point(self.grid_size, 0)
        bottomLeft = Point(0, self.grid_size)
        bottomRight = Point(self.grid_size, self.grid_size)
        pattern.add_crease(topLeft, topRight, CreaseType.BORDER)
        pattern.add_crease(topRight, bottomRight, CreaseType.BORDER)
        pattern.add_crease(bottomRight, bottomLeft, CreaseType.BORDER)
        pattern.add_crease(bottomLeft, topLeft, CreaseType.BORDER)

        return pattern

    def add_diagonal_crease(self, pattern: BoxPleatingPattern, start: Point) -> bool:
        return True
        # """Try to add a valid diagonal crease starting from the given point."""
        # possible_ends = []

        # # Calculate possible diagonal endpoints (45° angles)
        # x, y = start.x, start.y
        # directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

        # for dx, dy in directions:
        #     # Check all possible lengths of diagonal lines
        #     for length in range(1, self.grid_size + 1):
        #         new_x = x + dx * length
        #         new_y = y + dy * length

        #         # Check if endpoint is within grid
        #         if 0 <= new_x <= self.grid_size and 0 <= new_y <= self.grid_size:
        #             possible_ends.append(Point(new_x, new_y))

        # # Filter out endpoints that would create invalid patterns
        # valid_ends = []
        # for end in possible_ends:
        #     # Create a temporary copy of the pattern
        #     temp_pattern = BoxPleatingPattern(self.grid_size)
        #     temp_pattern.vertices = pattern.vertices.copy()
        #     temp_pattern.creases = pattern.creases.copy()

        #     # Try adding the crease
        #     crease_type = (
        #         CreaseType.MOUNTAIN if random.random() < 0.5 else CreaseType.VALLEY
        #     )
        #     if temp_pattern.add_crease(start, end, crease_type):
        #         # Check if pattern remains flat-foldable
        #         is_foldable, _ = temp_pattern.is_flat_foldable()
        #         if is_foldable:
        #             valid_ends.append((end, crease_type))

        # if valid_ends:
        #     # Choose a random valid endpoint
        #     end, crease_type = random.choice(valid_ends)
        #     pattern.add_crease(start, end, crease_type)
        #     return True

        # return False

    def generate_pattern(self, complexity: float = 0.5) -> BoxPleatingPattern:
        """
        Generate a complete flat-foldable box-pleating pattern.

        Args:
            complexity: Float between 0 and 1 determining how many additional
            creases to add beyond the base grid.
        """
        # Start with base grid
        pattern = self.generate_base_grid()

        # Calculate number of additional creases based on complexity
        max_additional = (self.grid_size + 1) * (self.grid_size + 1) // 2
        num_additional = int(max_additional * complexity)

        # Add diagonal creases
        attempts = 0
        added_creases = 0
        max_attempts = num_additional * 3  # Allow for some failed attempts

        while added_creases < num_additional and attempts < max_attempts:
            # Choose random starting point
            x = random.randint(0, self.grid_size)
            y = random.randint(0, self.grid_size)
            start = Point(x, y)

            if self.add_diagonal_crease(pattern, start):
                added_creases += 1

            attempts += 1

        # Clean up pattern by removing redundant vertices
        pattern.remove_redundant_vertices()

        return pattern


def main():
    # Example usage
    # generator = BoxPleatingGenerator(grid_size=4)
    # pattern = generator.generate_pattern(complexity=0.7)

    # Check if pattern is flat-foldable
    # is_foldable, violations = pattern.is_flat_foldable()
    # print(f"Generated pattern is flat-foldable: {is_foldable}")
    # if not is_foldable:
    #     print("Violations found:")
    #     for violation in violations:
    #         print(
    #             f"  At vertex ({violation['vertex']['x']}, {violation['vertex']['y']})"
    #         )

    # # Print pattern representation
    # print("\nPattern details:")
    # print(pattern)


    pattern = BoxPleatingPattern(4)

    # Add horizontal and vertical grid lines
    i = 1
    # Horizontal lines
    # start = Point(0, i)
    # end = Point(4, i)
    # pattern.add_crease(start, end, CreaseType.MOUNTAIN)
    # Vertical lines
    start = Point(i, 0)
    end = Point(i, 4)
    pattern.add_crease(start, end, CreaseType.VALLEY)
    pattern.add_crease(start, end, CreaseType.MOUNTAIN)

    # Add border creases
    topLeft = Point(0, 0)
    topRight = Point(4, 0)
    bottomLeft = Point(0, 4)
    bottomRight = Point(4,4)
    pattern.add_crease(topLeft, topRight, CreaseType.BORDER)
    pattern.add_crease(topRight, bottomRight, CreaseType.BORDER)
    pattern.add_crease(bottomRight, bottomLeft, CreaseType.BORDER)
    pattern.add_crease(bottomLeft, topLeft, CreaseType.BORDER)

    fc = FoldConverter()
    fc.save_fold(pattern, "pattern.fold")


if __name__ == "__main__":
    main()
