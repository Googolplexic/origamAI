"""
Box Pleating Pattern Module

This module provides tools for creating and validating box-pleating origami patterns.
Box pleating is a tessellation technique where all creases are at 45° or 90° angles
to each other, forming a grid-like pattern.
"""

from .models import Point, Crease, CreaseType
from .pattern import BoxPleatingPattern

__all__ = ["Point", "Crease", "CreaseType", "BoxPleatingPattern"]

__version__ = "1.0.0"
