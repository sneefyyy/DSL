import random
import copy
from .base import ExampleGenerator

class CreateShapeGenerator(ExampleGenerator):
    """
    A class to generate few-shot examples for the 'create shape' task.
    The patterns focus on value transformations and shape-within-shape creation.
    """
    
    def __init__(self, size=5):
        """
        Initialize the CreateShapeGenerator with a specified grid size.
        
        Args:
            size (int): The dimension of the square grid.
        """
        super().__init__(size)
    
    # create_shape.py - Updated create_shape method
    @staticmethod
    def create_shape(points, value, grid=None):
        """
        Create a shape on a grid at specified points with a given value.
        
        Args:
            points (list): List of (row, col) tuples where shape should be created
            value (int): Value to fill the shape with (1-9)
            grid (list): Optional existing grid to modify, creates new 5x5 grid if None
            
        Returns:
            list: Grid with the shape created
        """
        if grid is None:
            # Create empty 5x5 grid by default
            grid = [[0 for _ in range(5)] for _ in range(5)]
        else:
            # Deep copy to avoid modifying original
            grid = copy.deepcopy(grid)
        
        # Place shape on grid
        for row, col in points:
            if 0 <= row < len(grid) and 0 <= col < len(grid[0]):
                grid[row][col] = value
        
        return grid
    
    def detect_all_shapes(self, grid):
        """
        Detect all non-zero cells in the grid.
        
        Args:
            grid (list): 2D grid
            
        Returns:
            dict: Dictionary mapping values to their coordinates
        """
        shapes = {}
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] != 0:
                    value = grid[row][col]
                    if value not in shapes:
                        shapes[value] = []
                    shapes[value].append((row, col))
        return shapes
    
    def find_shape_bounds(self, points):
        """
        Find the bounding box of a shape.
        
        Args:
            points (list): List of (row, col) tuples
            
        Returns:
            tuple: (min_row, min_col, max_row, max_col)
        """
        if not points:
            return None
        min_row = min(p[0] for p in points)
        max_row = max(p[0] for p in points)
        min_col = min(p[1] for p in points)
        max_col = max(p[1] for p in points)
        return min_row, min_col, max_row, max_col
    
    def get_inner_cells(self, points):
        """
        Get cells that are inside a shape (surrounded by shape cells).
        
        Args:
            points (list): List of (row, col) tuples forming the shape
            
        Returns:
            list: List of (row, col) tuples that are inside the shape
        """
        if not points:
            return []
        
        min_row, min_col, max_row, max_col = self.find_shape_bounds(points)
        inner_cells = []
        
        # Check each cell within bounds
        for row in range(min_row + 1, max_row):
            for col in range(min_col + 1, max_col):
                if (row, col) not in points:
                    # Check if surrounded by shape cells
                    surrounded = True
                    for dr in [-1, 0, 1]:
                        if not surrounded:
                            break
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            neighbor = (row + dr, col + dc)
                            # Check if on boundary of bounding box
                            if (row + dr <= min_row or row + dr >= max_row or 
                                col + dc <= min_col or col + dc >= max_col):
                                if neighbor not in points:
                                    surrounded = False
                                    break
                    
                    if surrounded:
                        inner_cells.append((row, col))
        
        return inner_cells
    
    def generate_pattern_examples(self, pattern_type="increment_value"):
        """
        Generate examples following specific patterns.
        
        Args:
            pattern_type (str): Type of pattern to generate
            
        Returns:
            dict: Dictionary with train and test examples
        """
        if pattern_type == "increment_value":
            # Pattern: Change all shape values by +1
            # Train 1: Simple horizontal line
            value1 = random.randint(1, 8)
            shape1 = [(1, 1), (1, 2), (1, 3)]
            
            train1_input = self.create_empty_grid(self.size)
            train1_input = self.create_shape(shape1, value1, train1_input)
            
            train1_output = self.create_empty_grid(self.size)
            train1_output = self.create_shape(shape1, value1 + 1, train1_output)
            
            # Train 2: L-shape
            value2 = random.randint(1, 8)
            shape2 = [(2, 0), (3, 0), (3, 1)]
            
            train2_input = self.create_empty_grid(self.size)
            train2_input = self.create_shape(shape2, value2, train2_input)
            
            train2_output = self.create_empty_grid(self.size)
            train2_output = self.create_shape(shape2, value2 + 1, train2_output)
            
            # Test: Vertical line
            test_value = random.randint(1, 8)
            test_shape = [(0, 2), (1, 2), (2, 2)]
            
            test_input = self.create_empty_grid(self.size)
            test_input = self.create_shape(test_shape, test_value, test_input)
            
            test_output = self.create_empty_grid(self.size)
            test_output = self.create_shape(test_shape, test_value + 1, test_output)
            
            return {
                "train1": {"input": train1_input, "output": train1_output},
                "train2": {"input": train2_input, "output": train2_output},
                "test": {"input": test_input, "output": test_output,
                         "new_shape": test_shape, "new_value": test_value + 1}
            }
        
        elif pattern_type == "decrement_value":
            # Pattern: Change all shape values by -1
            # Train 1: Square
            value1 = random.randint(2, 9)
            shape1 = [(1, 1), (1, 2), (2, 1), (2, 2)]
            
            train1_input = self.create_empty_grid(self.size)
            train1_input = self.create_shape(shape1, value1, train1_input)
            
            train1_output = self.create_empty_grid(self.size)
            train1_output = self.create_shape(shape1, value1 - 1, train1_output)
            
            # Train 2: Plus sign
            value2 = random.randint(2, 9)
            shape2 = [(1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
            
            train2_input = self.create_empty_grid(self.size)
            train2_input = self.create_shape(shape2, value2, train2_input)
            
            train2_output = self.create_empty_grid(self.size)
            train2_output = self.create_shape(shape2, value2 - 1, train2_output)
            
            # Test: T-shape
            test_value = random.randint(2, 9)
            test_shape = [(0, 1), (0, 2), (0, 3), (1, 2)]
            
            test_input = self.create_empty_grid(self.size)
            test_input = self.create_shape(test_shape, test_value, test_input)
            
            test_output = self.create_empty_grid(self.size)
            test_output = self.create_shape(test_shape, test_value - 1, test_output)
            
            return {
                "train1": {"input": train1_input, "output": train1_output},
                "train2": {"input": train2_input, "output": train2_output},
                "test": {"input": test_input, "output": test_output,
                         "new_shape": test_shape, "new_value": test_value - 1}
            }
        
        elif pattern_type == "fill_inside":
            # Pattern: Fill inside of hollow shape with value+1
            # Train 1: Hollow square
            value1 = random.randint(1, 8)
            outer1 = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)]
            inner1 = [(2, 2)]
            
            train1_input = self.create_empty_grid(self.size)
            train1_input = self.create_shape(outer1, value1, train1_input)
            
            train1_output = copy.deepcopy(train1_input)
            train1_output = self.create_shape(inner1, value1 + 1, train1_output)
            
            # Train 2: Hollow rectangle
            value2 = random.randint(1, 8)
            outer2 = [(0, 1), (0, 2), (0, 3), (0, 4), 
                      (1, 1), (1, 4),
                      (2, 1), (2, 4),
                      (3, 1), (3, 2), (3, 3), (3, 4)]
            inner2 = [(1, 2), (1, 3), (2, 2), (2, 3)]
            
            train2_input = self.create_empty_grid(self.size)
            train2_input = self.create_shape(outer2, value2, train2_input)
            
            train2_output = copy.deepcopy(train2_input)
            train2_output = self.create_shape(inner2, value2 + 1, train2_output)
            
            # Test: Another hollow square
            test_value = random.randint(1, 8)
            test_outer = [(0, 0), (0, 1), (0, 2),
                          (1, 0), (1, 2),
                          (2, 0), (2, 1), (2, 2)]
            test_inner = [(1, 1)]
            
            test_input = self.create_empty_grid(self.size)
            test_input = self.create_shape(test_outer, test_value, test_input)
            
            test_output = copy.deepcopy(test_input)
            test_output = self.create_shape(test_inner, test_value + 1, test_output)
            
            return {
                "train1": {"input": train1_input, "output": train1_output},
                "train2": {"input": train2_input, "output": train2_output},
                "test": {"input": test_input, "output": test_output,
                         "new_shape": test_inner, "new_value": test_value + 1}
            }
        
        elif pattern_type == "double_value":
            # Pattern: Change shape value to 2x (capped at 9)
            # Train 1: Line
            value1 = random.randint(1, 4)
            shape1 = [(2, 1), (2, 2), (2, 3)]
            
            train1_input = self.create_empty_grid(self.size)
            train1_input = self.create_shape(shape1, value1, train1_input)
            
            train1_output = self.create_empty_grid(self.size)
            train1_output = self.create_shape(shape1, min(value1 * 2, 9), train1_output)
            
            # Train 2: Corner shape
            value2 = random.randint(1, 4)
            shape2 = [(0, 0), (0, 1), (1, 0)]
            
            train2_input = self.create_empty_grid(self.size)
            train2_input = self.create_shape(shape2, value2, train2_input)
            
            train2_output = self.create_empty_grid(self.size)
            train2_output = self.create_shape(shape2, min(value2 * 2, 9), train2_output)
            
            # Test: Small square
            test_value = random.randint(1, 4)
            test_shape = [(3, 3), (3, 4), (4, 3), (4, 4)]
            
            test_input = self.create_empty_grid(self.size)
            test_input = self.create_shape(test_shape, test_value, test_input)
            
            test_output = self.create_empty_grid(self.size)
            test_output = self.create_shape(test_shape, min(test_value * 2, 9), test_output)
            
            return {
                "train1": {"input": train1_input, "output": train1_output},
                "train2": {"input": train2_input, "output": train2_output},
                "test": {"input": test_input, "output": test_output,
                         "new_shape": test_shape, "new_value": min(test_value * 2, 9)}
            }
    
    def create_fewshot_examples(self, num_examples=100):
        """
        Generate multiple few-shot examples.
        
        Args:
            num_examples (int): Number of examples to generate
            
        Returns:
            list: List of few-shot examples
        """
        examples = []
        pattern_types = ["increment_value", "decrement_value", "fill_inside", "double_value"]
        
        for i in range(num_examples):
            pattern_type = pattern_types[i % len(pattern_types)]
            pattern_data = self.generate_pattern_examples(pattern_type)
            
            # Create solution based on pattern type
            # In create_fewshot_examples method:
            if pattern_type in ["increment_value", "decrement_value", "double_value"]:
                # For value transformation patterns - create new grid
                solution = [
                    f"output_grid = CreateShapeGenerator.create_shape({pattern_data['test']['new_shape']}, {pattern_data['test']['new_value']})"
                ]
            else:  # fill_inside
                # For filling patterns - modify existing grid
                solution = [
                    f"output_grid = CreateShapeGenerator.create_shape({pattern_data['test']['new_shape']}, {pattern_data['test']['new_value']}, test_input)"
                ]
            
            examples.append({
                "train_input1": pattern_data["train1"]["input"],
                "train_output1": pattern_data["train1"]["output"],
                "train_input2": pattern_data["train2"]["input"],
                "train_output2": pattern_data["train2"]["output"],
                "test_input": pattern_data["test"]["input"],
                "test_output": pattern_data["test"]["output"],
                "solution": solution
            })
        
        return examples
    