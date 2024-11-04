import unittest
from bp_pattern_class import BoxPleatingPattern, Point, CreaseType


class TestBoxPleatingPattern(unittest.TestCase):
    def setUp(self):
        self.pattern = BoxPleatingPattern(grid_size=10)  # Increased grid size

    def test_horizontal_overlap_rejection(self):
        """Test that overlapping horizontal segments are rejected"""
        # Create initial horizontal crease - single crease first
        result = self.pattern.add_crease(
            Point(1.0, 1.0), Point(2.0, 1.0), CreaseType.VALLEY
        )
        self.assertTrue(result, "Should accept initial horizontal crease")

        # Try to add overlapping crease
        result = self.pattern.add_crease(
            Point(1.0, 1.0), Point(2.0, 1.0), CreaseType.VALLEY
        )
        self.assertFalse(result, "Should reject completely overlapping crease")

        # Try to add partially overlapping crease
        result = self.pattern.add_crease(
            Point(1.5, 1.0), Point(2.5, 1.0), CreaseType.VALLEY
        )
        self.assertFalse(result, "Should reject partially overlapping crease")

    def test_vertical_overlap_rejection(self):
        """Test that overlapping vertical segments are rejected"""
        # Create initial vertical crease
        result = self.pattern.add_crease(
            Point(1.0, 1.0), Point(1.0, 2.0), CreaseType.VALLEY
        )
        self.assertTrue(result, "Should accept initial vertical crease")

        # Try to add overlapping vertical crease
        result = self.pattern.add_crease(
            Point(1.0, 1.0), Point(1.0, 2.0), CreaseType.VALLEY
        )
        self.assertFalse(result, "Should reject overlapping vertical crease")

    def test_diagonal_overlap_rejection(self):
        """Test that overlapping diagonal segments are rejected"""
        # Create initial diagonal crease - make sure it's a valid box-pleating diagonal
        result = self.pattern.add_crease(
            Point(1.0, 1.0), Point(2.0, 2.0), CreaseType.VALLEY
        )
        self.assertTrue(result, "Should accept initial diagonal crease")

        # Try to add overlapping diagonal crease
        result = self.pattern.add_crease(
            Point(1.0, 1.0), Point(2.0, 2.0), CreaseType.VALLEY
        )
        self.assertFalse(result, "Should reject overlapping diagonal crease")

    def test_non_overlapping_parallel_acceptance(self):
        """Test that parallel but non-overlapping segments are accepted"""
        # Create initial horizontal crease
        result = self.pattern.add_crease(
            Point(1.0, 1.0), Point(2.0, 1.0), CreaseType.VALLEY
        )
        self.assertTrue(result, "Should accept initial horizontal crease")

        # Add parallel but non-overlapping crease
        result = self.pattern.add_crease(
            Point(3.0, 1.0), Point(4.0, 1.0), CreaseType.VALLEY
        )
        self.assertTrue(result, "Should accept parallel non-overlapping crease")

    def test_sequential_creases(self):
        """Test that sequential creases are accepted"""
        # Add first crease
        result = self.pattern.add_crease(
            Point(1.0, 1.0), Point(2.0, 1.0), CreaseType.VALLEY
        )
        self.assertTrue(result, "Should accept first crease")

        # Add sequential crease
        result = self.pattern.add_crease(
            Point(2.0, 1.0), Point(3.0, 1.0), CreaseType.VALLEY
        )
        self.assertTrue(result, "Should accept sequential crease")

    def test_floating_point_precision(self):
        """Test handling of floating point precision in overlap detection"""
        # Create initial crease
        result = self.pattern.add_crease(
            Point(1.0, 1.0), Point(2.0, 1.0), CreaseType.VALLEY
        )
        self.assertTrue(result, "Should accept initial crease")

        # Try to add slightly offset parallel crease
        result = self.pattern.add_crease(
            Point(1.0, 1.0 + 1e-10), Point(2.0, 1.0 + 1e-10), CreaseType.VALLEY
        )
        self.assertFalse(result, "Should reject nearly overlapping crease")


def run_tests():
    unittest.main(argv=[""], exit=False)


if __name__ == "__main__":
    run_tests()
