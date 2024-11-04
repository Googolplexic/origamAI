#!/usr/bin/env python3
"""
Script to analyze a FOLD format origami pattern for flat-foldability.

This script:
1. Loads a FOLD format file
2. Converts it to a box-pleating pattern
3. Removes redundant vertices
4. Checks for flat-foldability
5. Outputs the results and saves to a new FOLD file
"""

from box_pleating import BoxPleatingPattern as bp
from box_pleating.fold import FoldConverter as fc
import json
import sys


def load_fold_file(file_name: str) -> dict:
    """
    Load and parse a FOLD format file.

    Args:
        file_name: Path to the FOLD file

    Returns:
        dict: Parsed FOLD data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file isn't valid JSON
    """
    try:
        with open(file_name, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: The file '{file_name}' is not a valid JSON file.")
        sys.exit(1)


def print_foldability_results(is_foldable: bool, violations: list) -> None:
    """
    Print the results of the flat-foldability check.

    Args:
        is_foldable: Whether the pattern is flat-foldable
        violations: List of violations if not flat-foldable
    """
    if is_foldable:
        print("The pattern is flat foldable.")
    else:
        print("The pattern is not flat foldable.")
        print("Reasons:")
        for violation in violations:
            print(json.dumps(violation, indent=2))


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_pattern.py <input_fold_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = "output.fold"

    # Load and process the pattern
    fold_data = load_fold_file(input_file)
    converter = fc()

    # Convert to box-pleating pattern and clean up
    pattern = converter.from_fold(fold_data)
    pattern.remove_redundant_vertices()

    # Check flat-foldability
    is_flat_foldable, violations = pattern.is_flat_foldable()
    print_foldability_results(is_flat_foldable, violations)

    # Save the processed pattern
    converter.save_fold(pattern, output_file)
    print(f"Processed pattern saved to {output_file}")


if __name__ == "__main__":
    main()
