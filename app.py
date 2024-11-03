from bp_pattern_class import BoxPleatingPattern as bp
from fold_convert_class import FoldConverter as fc
import json

if __name__ == "__main__":
    file_name = "input.fold"
    try:
        with open(file_name, "r") as file:
            fold_data = json.load(file)
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: The file '{file_name}' is not a valid JSON file.")
        exit(1)
    pattern = fc.from_fold(fold_data)
    pattern.remove_redundant_vertices()
    is_flat_foldable = bp.is_flat_foldable(pattern)
    if is_flat_foldable[0]:
        print("The pattern is flat foldable.")
    else:
        print("The pattern is not flat foldable.")
        print("Reasons:")
        for violation in is_flat_foldable[1]:
            print(json.dumps(violation, indent=2))
    fc.to_fold(pattern, "output.fold")
