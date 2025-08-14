# interpreter.py (copied from repository root)
import json
import random
import copy
import os
import sys
from pathlib import Path
import inspect

# Ensure the repository root is on sys.path so top-level packages like
# `generators/` (located at the workspace root) can be imported when running
# modules from inside `src/`.
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
	sys.path.insert(0, str(repo_root))

# Note: we import `generators` dynamically inside methods after adjusting
# sys.path. Avoid top-level imports which can fail when the module search
# path doesn't include the workspace root.

class ExampleInterpreter:
	"""
	Interprets and tests JSON examples against their corresponding generator classes.
	"""
    
	def __init__(self, json_dir="JSON_training"):
		"""
		Initialize the interpreter with directory for JSON files.
        
		Args:
			json_dir (str): Directory containing JSON training files
		"""
		self.json_dir = json_dir
		self.generator_classes = self.discover_generator_classes()
    
	def discover_generator_classes(self):
		"""
		Discover all available generator classes from the generators package.
        
		Returns:
			dict: Mapping of class names to class objects
		"""
		generator_classes = {}
        
		# Get all items from generators module
		import generators
		for name in dir(generators):
			obj = getattr(generators, name)
			if inspect.isclass(obj) and name.endswith('Generator'):
				generator_classes[name] = obj
        
		return generator_classes
    
	def load_json_examples(self, filename):
		"""
		Load examples from a JSON file.
        
		Args:
			filename (str): Name of the JSON file
            
		Returns:
			list: List of examples from the JSON file
		"""
		filepath = os.path.join(self.json_dir, filename)
		with open(filepath, 'r') as f:
			return json.load(f)
    
	def create_execution_namespace(self):
		"""
		Create a namespace with all necessary imports and classes for execution.
        
		Returns:
			dict: Namespace dictionary with all available classes and utilities
		"""
		namespace = {
			'copy': copy,
			'__builtins__': __builtins__,
		}
        
		# Add all discovered generator classes
		namespace.update(self.generator_classes)
        
		# Create instances for non-base generators
		for class_name, class_obj in self.generator_classes.items():
			if class_name != 'ExampleGenerator':
				try:
					instance = class_obj()
					# Store with a lowercase name for convenience
					instance_name = class_name[0].lower() + class_name[1:]
					namespace[instance_name] = instance
				except:
					pass

		# Add direct function aliases for common method names so generated code
		# that uses slightly different naming (e.g. `Rotate_shape_on_grid` or
		# `flood_fill`) will still resolve to the correct implementation.
		for class_name, class_obj in self.generator_classes.items():
			try:
				for attr in dir(class_obj):
					if attr.startswith('_'):
						continue
					try:
						attr_obj = getattr(class_obj, attr)
						if callable(attr_obj):
							# Add the raw method name as a top-level name
							if attr not in namespace:
								namespace[attr] = attr_obj

							# Add a variant where the first token is capitalized
							# e.g. rotate_shape_on_grid -> Rotate_shape_on_grid
							parts = attr.split('_')
							if len(parts) > 1:
								cap_variant = '_'.join([parts[0].capitalize()] + parts[1:])
								if cap_variant not in namespace:
									namespace[cap_variant] = attr_obj
					except Exception:
						# ignore attribute inspection failures
						continue
			except Exception:
				continue
        
		return namespace
    
	def test_example(self, example):
		"""
		Test a single example by executing its solution.
        
		Args:
			example (dict): Example containing train and test data
            
		Returns:
			tuple: (success, computed_output)
		"""
		test_input = example['test_input']
		expected_output = example['test_output']
		solution_lines = example['solution']
        
		# Create namespace with all generators pre-loaded
		namespace = self.create_execution_namespace()
        
		# IMPORTANT: Create a deep copy of test_input to avoid mutations
		namespace['test_input'] = copy.deepcopy(test_input)
        
		try:
			# Join all solution lines into a single code block
			full_solution = '\n'.join(solution_lines)
            
			# Execute the entire solution as one block
			exec(full_solution, namespace)
            
			# Get output
			output_grid = namespace.get('output_grid', namespace.get('grid', None))
            
			if output_grid is None:
				return False, None
            
			# Compare
			success = output_grid == expected_output
			return success, output_grid
            
		except Exception as e:
			print(f"Error executing solution: {e}")
			print(f"Solution:")
			for line in solution_lines:
				print(f"  {line}")
			import traceback
			traceback.print_exc()
			return False, None
        
	def print_grid_comparison(self, grid1, grid2, title1="Grid 1", title2="Grid 2"):
		"""
		Print two grids side by side for comparison.
		"""
		print(f"\n{title1:<20} {title2:<20}")
		print("-" * 41)
        
		max_rows = max(len(grid1), len(grid2)) if grid1 and grid2 else 0
		for i in range(max_rows):
			# Row from grid1
			if grid1 and i < len(grid1):
				row1 = " ".join(str(cell) if cell != 0 else "." for cell in grid1[i])
			else:
				row1 = ""
            
			# Row from grid2
			if grid2 and i < len(grid2):
				row2 = " ".join(str(cell) if cell != 0 else "." for cell in grid2[i])
			else:
				row2 = ""
            
			print(f"{row1:<20} {row2:<20}")
    
	def test_json_file(self, json_filename, num_tests=5, verbose=True):
		"""
		Test random examples from a JSON file.
        
		Args:
			json_filename (str): Name of the JSON file to test
			num_tests (int): Number of random examples to test
			verbose (bool): Whether to print detailed output
            
		Returns:
			tuple: (passed, total) number of tests
		"""
		try:
			# Load examples
			examples = self.load_json_examples(json_filename)
            
			# Select random examples
			test_examples = random.sample(examples, min(num_tests, len(examples)))
            
			passed = 0
			total = len(test_examples)
            
			print(f"\nTesting {json_filename}")
			print("=" * 60)
            
			for i, example in enumerate(test_examples):
				if verbose:
					print(f"\nTest {i+1}/{total}")
                
				# Test the example
				success, computed_output = self.test_example(example)
                
				if success:
					passed += 1
					if verbose:
						print("✓ PASSED")
				else:
					if verbose:
						print("✗ FAILED")
						# Show the grids for failed tests
						test_input = example['test_input']
						expected_output = example['test_output']
                        
						print("\nSolution:")
						for line in example['solution']:
							print(f"  {line}")
                        
						self.print_grid_comparison(test_input, expected_output, "Test Input", "Expected Output")
                        
						if computed_output is not None:
							self.print_grid_comparison(expected_output, computed_output, "Expected", "Computed")
						else:
							print("\nNo output was computed (output_grid not found)")
            
			print(f"\nResults: {passed}/{total} tests passed")
			return passed, total
            
		except Exception as e:
			print(f"Error testing {json_filename}: {e}")
			import traceback
			traceback.print_exc()
			return 0, 0
        
	def test_all_json_files(self, num_tests_per_file=5):
		"""
		Test all JSON files in the JSON directory.
        
		Args:
			num_tests_per_file (int): Number of tests to run per JSON file
		"""
		json_files = [f for f in os.listdir(self.json_dir) if f.endswith('.json')]
        
		total_passed = 0
		total_tests = 0
        
		for json_file in json_files:
			passed, total = self.test_json_file(json_file, num_tests_per_file)
			total_passed += passed
			total_tests += total
        
		print(f"\n{'='*60}")
		print(f"OVERALL RESULTS: {total_passed}/{total_tests} tests passed")
		if total_tests > 0:
			print(f"Success rate: {total_passed/total_tests*100:.1f}%")


def main():
	"""
	Main function to run the interpreter tests.
	"""
	interpreter = ExampleInterpreter()
    
	# Test specific file
	# print("Testing MoveShape.json examples...")
	# interpreter.test_json_file("MoveShape.json", num_tests=5, verbose=True)
    
	# Uncomment to test all JSON files
	print("\nTesting all JSON files...")
	interpreter.test_all_json_files(num_tests_per_file=3)


if __name__ == "__main__":
	main()

