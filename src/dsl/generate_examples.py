import os
# Import the ConnectSegmentGenerator class
from generators.move_shape import MoveShapeGenerator
from generators.connect import ConnectGenerator
from generators.rotate_shape import RotateShapeGenerator
from generators.create_shape import CreateShapeGenerator
from generators.flood_fill import FloodFillGenerator
from generators.mirror_shape import MirrorShapeGenerator
from generators.symmetry_complete_generator import SymmetryCompleteGenerator
from generators.extract_pattern_generator import ExtractPatternGenerator
from generators.repeat_pattern_generator import RepeatPatternGenerator
from generators.count_and_transform_generator import CountAndTransformGenerator

if __name__ == "__main__":
	json_training_dir = "JSON_training/"
	os.makedirs(json_training_dir, exist_ok=True)

	generator = ConnectGenerator(size=5)
	generator.save_fewshot_examples(os.path.join(json_training_dir, "Connect.json"), num_examples=4000)

	generator = MoveShapeGenerator(size=5)
	generator.save_fewshot_examples(os.path.join(json_training_dir, "MoveShape.json"), num_examples=4000)

	# Generate RotateShape examples
	generator = RotateShapeGenerator(size=5)
	generator.save_fewshot_examples(os.path.join(json_training_dir, "RotateShape.json"), num_examples=4000)

	generator = CreateShapeGenerator(size=5)
	generator.save_fewshot_examples(os.path.join(json_training_dir, "CreateShape.json"), num_examples=4000)

	generator = FloodFillGenerator(size=5)
	generator.save_fewshot_examples(os.path.join(json_training_dir, "FloodFill.json"), num_examples=4000)

	generator = MirrorShapeGenerator(size=5)
	generator.save_fewshot_examples(os.path.join(json_training_dir, "MirrorShape.json"), num_examples=4000)

	generator = SymmetryCompleteGenerator(size=5)
	generator.save_fewshot_examples(os.path.join(json_training_dir, "SymmetryComplete.json"), num_examples=4000)

	generator = ExtractPatternGenerator(size=5)
	generator.save_fewshot_examples(os.path.join(json_training_dir, "ExtractPattern.json"), num_examples=4000)

	generator = RepeatPatternGenerator(size=5)
	generator.save_fewshot_examples(os.path.join(json_training_dir, "RepeatPattern.json"), num_examples=4000)

	generator = CountAndTransformGenerator(size=5)
	generator.save_fewshot_examples(os.path.join(json_training_dir, "CountAndTransform.json"), num_examples=4000)

	# You can also generate and inspect a single example directly:
	# single_example = generator.generate_connect_example()
	# print("\n--- Single Generated Example ---")
	# print(json.dumps(single_example, indent=2))

	# Or generate a list of examples without saving them to a file:
	# examples_list = generator.create_fewshot_examples(num_examples=3)
	# print("\n--- First example from a generated list (not saved to file) ---")
	# print(json.dumps(examples_list[0], indent=2))

