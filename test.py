import unittest
from bp_pattern_class import BoxPleatingPattern, Point, CreaseType, Crease

class TestBoxPleatingPattern(unittest.TestCase):
    def setUp(self):
        """Set up a fresh pattern for each test"""
        self.pattern = BoxPleatingPattern(grid_size=4)
        
    def test_basic_crease_creation(self):
        """Test basic crease creation and validation"""
        # Valid creases
        self.assertTrue(self.pattern.add_crease(Point(0, 0), Point(4, 0), CreaseType.MOUNTAIN))  # Horizontal
        self.assertTrue(self.pattern.add_crease(Point(0, 0), Point(0, 4), CreaseType.VALLEY))   # Vertical
        self.assertTrue(self.pattern.add_crease(Point(0, 0), Point(4, 4), CreaseType.MOUNTAIN)) # Diagonal
        
        # Invalid creases
        self.assertFalse(self.pattern.add_crease(Point(-1, 0), Point(4, 0), CreaseType.MOUNTAIN))  # Out of grid
        self.assertFalse(self.pattern.add_crease(Point(0, 0), Point(3, 2), CreaseType.MOUNTAIN))   # Invalid angle

    def test_crease_intersection(self):
        """Test crease intersection handling"""
        # Create intersecting creases
        self.pattern.add_crease(Point(0, 2), Point(4, 2), CreaseType.MOUNTAIN)  # Horizontal
        self.pattern.add_crease(Point(2, 0), Point(2, 4), CreaseType.VALLEY)    # Vertical
        
        # Check that intersection point was created as vertex
        self.assertIn(Point(2, 2), self.pattern.vertices)
        
        # Verify creases were split
        intersection_creases = [c for c in self.pattern.creases if c.start == Point(2, 2) or c.end == Point(2, 2)]
        self.assertEqual(len(intersection_creases), 4)  # Should have 4 creases meeting at intersection

    def test_multiple_intersections(self):
        """Test handling of multiple intersections"""
        # Create a pattern with multiple intersections
        self.pattern.add_crease(Point(0, 2), Point(4, 2), CreaseType.MOUNTAIN)  # Horizontal
        self.pattern.add_crease(Point(1, 0), Point(1, 4), CreaseType.VALLEY)    # Vertical 1
        self.pattern.add_crease(Point(3, 0), Point(3, 4), CreaseType.VALLEY)    # Vertical 2
        
        # Check both intersection points were created
        self.assertIn(Point(1, 2), self.pattern.vertices)
        self.assertIn(Point(3, 2), self.pattern.vertices)
        
        # Verify correct number of creases
        self.assertEqual(len(self.pattern.creases), 6)  # Original 3 creases split into 6

    def test_overlapping_creases(self):
        """Test handling of overlapping creases"""
        # Create initial crease
        self.assertTrue(self.pattern.add_crease(Point(0, 0), Point(4, 0), CreaseType.MOUNTAIN))
        
        # Try to add overlapping crease of same type
        result = self.pattern.add_crease(Point(0, 0), Point(4, 0), CreaseType.MOUNTAIN)
        self.assertTrue(result)  # Should succeed but not create duplicate
        self.assertEqual(len(self.pattern.creases), 1)  # Should still only have one crease
        
        # Try to add overlapping crease of different type
        result = self.pattern.add_crease(Point(0, 0), Point(4, 0), CreaseType.VALLEY)
        self.assertFalse(result)  # Should fail
        self.assertEqual(len(self.pattern.creases), 1)  # Should not add new crease

    def test_partial_overlap(self):
        """Test handling of partially overlapping creases"""
        # Create initial crease
        self.pattern.add_crease(Point(0, 0), Point(4, 0), CreaseType.MOUNTAIN)
        
        # Try to add partially overlapping crease of same type
        result = self.pattern.add_crease(Point(2, 0), Point(4, 0), CreaseType.MOUNTAIN)
        self.assertTrue(result)  # Should succeed but not create duplicate
        self.assertEqual(len(self.pattern.creases), 1)  # Should still only have one crease
        
        # Try to add partially overlapping crease of different type
        result = self.pattern.add_crease(Point(2, 0), Point(4, 0), CreaseType.VALLEY)
        self.assertFalse(result)  # Should fail
        
    def test_complex_intersection_pattern(self):
        """Test a complex pattern with multiple intersections and overlaps"""
        # Create a grid of creases
        # Horizontal creases
        self.pattern.add_crease(Point(0, 1), Point(4, 1), CreaseType.MOUNTAIN)
        self.pattern.add_crease(Point(0, 3), Point(4, 3), CreaseType.MOUNTAIN)
        
        # Vertical creases
        self.pattern.add_crease(Point(1, 0), Point(1, 4), CreaseType.VALLEY)
        self.pattern.add_crease(Point(3, 0), Point(3, 4), CreaseType.VALLEY)
        
        # Diagonal crease through multiple intersections
        self.pattern.add_crease(Point(0, 0), Point(4, 4), CreaseType.MOUNTAIN)
        
        # Verify intersection points
        expected_intersections = {
            Point(1, 1), Point(3, 3),  # Diagonal with horizontal/vertical
            Point(1, 3), Point(3, 1),  # Diagonal with horizontal/vertical
            Point(1, 1), Point(1, 3),  # Vertical with horizontal
            Point(3, 1), Point(3, 3)   # Vertical with horizontal
        }
        
        for point in expected_intersections:
            self.assertIn(point, self.pattern.vertices)
            
    def test_grid_alignment(self):
        """Test that all vertices remain grid-aligned after operations"""
        # Create a complex pattern
        self.pattern.add_crease(Point(0, 2), Point(4, 2), CreaseType.MOUNTAIN)
        self.pattern.add_crease(Point(0, 0), Point(4, 4), CreaseType.VALLEY)
        
        # Check all vertices are at integer coordinates
        for vertex in self.pattern.vertices:
            self.assertEqual(vertex.x, round(vertex.x))
            self.assertEqual(vertex.y, round(vertex.y))
            
        # Check all crease endpoints are at integer coordinates
        for crease in self.pattern.creases:
            self.assertEqual(crease.start.x, round(crease.start.x))
            self.assertEqual(crease.start.y, round(crease.start.y))
            self.assertEqual(crease.end.x, round(crease.end.x))
            self.assertEqual(crease.end.y, round(crease.end.y))

    def test_crease_through_vertex(self):
        """Test adding a crease that passes through an existing vertex"""
        # Create initial creases forming a vertex
        self.pattern.add_crease(Point(0, 2), Point(4, 2), CreaseType.MOUNTAIN)
        self.pattern.add_crease(Point(2, 0), Point(2, 4), CreaseType.VALLEY)
        
        # Add a diagonal crease through the intersection point
        self.pattern.add_crease(Point(0, 0), Point(4, 4), CreaseType.MOUNTAIN)
        
        # Verify the vertex at (2,2) is connected to 6 crease segments
        vertex_creases = [c for c in self.pattern.creases if c.start == Point(2, 2) or c.end == Point(2, 2)]
        self.assertEqual(len(vertex_creases), 6)

    def test_crease_endpoints_on_existing_creases(self):
        """Test adding a crease with endpoints on existing creases"""
        # Create a rectangular frame
        self.pattern.add_crease(Point(0, 0), Point(4, 0), CreaseType.MOUNTAIN)  # Bottom
        self.pattern.add_crease(Point(4, 0), Point(4, 4), CreaseType.MOUNTAIN)  # Right
        
        # Add a crease with endpoints on existing creases
        self.pattern.add_crease(Point(2, 0), Point(4, 2), CreaseType.VALLEY)
        
        # Verify the new endpoints were properly handled
        self.assertIn(Point(2, 0), self.pattern.vertices)
        self.assertIn(Point(4, 2), self.pattern.vertices)

if __name__ == '__main__':
    unittest.main()