import random
import copy
from .base import ExampleGenerator

class MoveShapeGenerator(ExampleGenerator):
    """
    A class to generate few-shot examples for the 'move shape' task.
    It encapsulates the logic for detecting shapes at given centers, moving them,
    and structuring the training and testing examples. Inherits from ExampleGenerator.
    """

    def __init__(self, size=10):
        """
        Initializes the MoveShape with a specified grid size,
        calling the base class constructor.

        Args:
            size (int): The default dimension of the square grid for generated examples.
        """
        super().__init__(size)

    @staticmethod
    def detect_shape_at_center(grid, center_row, center_col):
        """
        Detects a contiguous shape in the grid starting from the given center coordinates.

        Args:
            grid (list): 2D list representing the grid.
            center_row (int): Row coordinate of the shape's center.
            center_col (int): Column coordinate of the shape's center.

        Returns:
            tuple: (shape_coords, value) where shape_coords is a list of (row, col) tuples
                   and value is the shape's color/value. Returns ([], 0) if no shape found at center.
        """
        if center_row < 0 or center_row >= len(grid) or center_col < 0 or center_col >= len(grid[0]):
            return ([], 0)
        
        value = grid[center_row][center_col]
        if value == 0:
            return ([], 0)
        
        shape_coords = []
        visited = set()
        stack = [(center_row, center_col)]
        
        while stack:
            row, col = stack.pop()
            if (row, col) in visited:
                continue
            if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]):
                continue
            if grid[row][col] != value:
                continue
                
            visited.add((row, col))
            shape_coords.append((row, col))
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                stack.append((row + dr, col + dc))
        
        return (sorted(shape_coords), value)

    @staticmethod
    def get_shape_bounds(shape_coords):
        """
        Gets the bounding box of a shape.

        Args:
            shape_coords (list): List of (row, col) tuples representing the shape.

        Returns:
            tuple: (min_row, min_col, max_row, max_col)
        """
        if not shape_coords:
            return (0, 0, 0, 0)
        
        rows = [coord[0] for coord in shape_coords]
        cols = [coord[1] for coord in shape_coords]
        
        return (min(rows), min(cols), max(rows), max(cols))

    def is_valid_move(self, shape_coords, delta_row, delta_col, grid_size):
        """
        Checks if moving a shape by the given deltas keeps it within the grid bounds.

        Args:
            shape_coords (list): List of (row, col) tuples representing the shape.
            delta_row (int): Row offset for movement.
            delta_col (int): Column offset for movement.
            grid_size (int): Size of the grid.

        Returns:
            bool: True if the move is valid, False otherwise.
        """
        for row, col in shape_coords:
            new_row = row + delta_row
            new_col = col + delta_col
            if new_row < 0 or new_row >= grid_size or new_col < 0 or new_col >= grid_size:
                return False
        return True

    @staticmethod
    def move_shape(grid, center_row, center_col, delta_row, delta_col):
        """
        Moves a shape centered at the given coordinates by the specified deltas.

        Args:
            grid (list): 2D list representing the input grid.
            center_row (int): Row coordinate of the shape's center.
            center_col (int): Column coordinate of the shape's center.
            delta_row (int): Row offset for movement.
            delta_col (int): Column offset for movement.

        Returns:
            list: New grid with the shape moved.
        """
        shape_coords, value = MoveShapeGenerator.detect_shape_at_center(grid, center_row, center_col)
        if not shape_coords:
            return grid
        
        # Check if move is valid
        for row, col in shape_coords:
            new_row = row + delta_row
            new_col = col + delta_col
            if new_row < 0 or new_row >= len(grid) or new_col < 0 or new_col >= len(grid[0]):
                return grid
        
        # Create new grid
        new_grid = [[0 for _ in range(len(grid[0]))] for _ in range(len(grid))]
        
        # Move shape
        for row, col in shape_coords:
            new_row = row + delta_row
            new_col = col + delta_col
            new_grid[new_row][new_col] = value
        
        return new_grid

    def generate_random_shape(self, max_size=4):
        """
        Generates a random connected shape.

        Args:
            max_size (int): Maximum dimension of the shape.

        Returns:
            tuple: (shape_coords, center_row, center_col) - List of (row, col) tuples and the shape's center.
        """
        # Start with a single cell
        shape = [(0, 0)]
        shape_size = random.randint(2, min(max_size * max_size, 9))
        
        while len(shape) < shape_size:
            # Pick a random cell from the shape
            base_cell = random.choice(shape)
            # Try to add a neighbor
            for _ in range(10):  # Try up to 10 times
                dr, dc = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                new_cell = (base_cell[0] + dr, base_cell[1] + dc)
                if new_cell not in shape:
                    shape.append(new_cell)
                    break
        
        # Calculate center (average of all coordinates)
        avg_row = sum(coord[0] for coord in shape) / len(shape)
        avg_col = sum(coord[1] for coord in shape) / len(shape)
        
        # Find the shape cell closest to the average
        min_dist = float('inf')
        center_row, center_col = shape[0]
        for r, c in shape:
            dist = abs(r - avg_row) + abs(c - avg_col)
            if dist < min_dist:
                min_dist = dist
                center_row, center_col = r, c
        
        return shape, center_row, center_col

    def generate_move_shape_example(self):
        """
        Generates a single 'move shape' example, including the input grid,
        the expected output grid (with the shape moved), and the movement parameters.

        Returns:
            dict: A dictionary containing the generated example data.
        """
        # Generate a random shape with center
        shape_template, template_center_r, template_center_c = self.generate_random_shape()
        
        # Get shape bounds to determine valid placement
        min_r, min_c, max_r, max_c = self.get_shape_bounds(shape_template)
        shape_height = max_r - min_r + 1
        shape_width = max_c - min_c + 1
        
        # Place shape at a random valid position
        max_start_row = self.size - shape_height
        max_start_col = self.size - shape_width
        
        if max_start_row <= 0 or max_start_col <= 0:
            # Shape too big for grid, use smaller grid or shape
            return self.generate_move_shape_example()
        
        start_row = random.randint(0, max_start_row)
        start_col = random.randint(0, max_start_col)
        
        # Create input grid with shape
        input_grid = self.create_empty_grid(self.size)
        value = random.randint(1, 9)
        
        actual_shape = []
        for r, c in shape_template:
            actual_row = start_row + r - min_r
            actual_col = start_col + c - min_c
            input_grid[actual_row][actual_col] = value
            actual_shape.append((actual_row, actual_col))
        
        # Calculate actual center coordinates
        actual_center_row = start_row + template_center_r - min_r
        actual_center_col = start_col + template_center_c - min_c
        
        # Generate random valid movement
        while True:
            delta_row = random.randint(-3, 3)
            delta_col = random.randint(-3, 3)
            if delta_row == 0 and delta_col == 0:
                continue
            if self.is_valid_move(actual_shape, delta_row, delta_col, self.size):
                break
        
        # Create output grid
        output_grid = self.move_shape(input_grid, actual_center_row, actual_center_col, delta_row, delta_col)
        
        return {
            "input": input_grid,
            "output": output_grid,
            "center_row": actual_center_row,
            "center_col": actual_center_col,
            "delta_row": delta_row,
            "delta_col": delta_col,
            "value": value
        }

    def create_fewshot_examples(self, num_examples=100):
        """
        Generates a list of multiple few-shot examples, each containing
        two training pairs and one testing pair with a corresponding solution.
        All examples use the same movement pattern.

        Args:
            num_examples (int): The number of examples to generate.

        Returns:
            list: A list of dictionaries, where each dictionary is a few-shot example.
        """
        examples = []
        for i in range(num_examples):
            # Generate first training example
            train1 = self.generate_move_shape_example()
            
            # Generate second training example with different shape/position but same movement
            while True:
                # Generate a random shape with center
                shape_template, template_center_r, template_center_c = self.generate_random_shape()
                
                # Get shape bounds to determine valid placement
                min_r, min_c, max_r, max_c = self.get_shape_bounds(shape_template)
                shape_height = max_r - min_r + 1
                shape_width = max_c - min_c + 1
                
                # Place shape at a random valid position
                max_start_row = self.size - shape_height
                max_start_col = self.size - shape_width
                
                if max_start_row <= 0 or max_start_col <= 0:
                    continue
                
                start_row = random.randint(0, max_start_row)
                start_col = random.randint(0, max_start_col)
                
                # Create train2 input grid with shape
                train2_input_grid = self.create_empty_grid(self.size)
                train2_value = random.randint(1, 9)
                
                actual_shape = []
                for r, c in shape_template:
                    actual_row = start_row + r - min_r
                    actual_col = start_col + c - min_c
                    train2_input_grid[actual_row][actual_col] = train2_value
                    actual_shape.append((actual_row, actual_col))
                
                # Calculate actual center coordinates
                train2_center_row = start_row + template_center_r - min_r
                train2_center_col = start_col + template_center_c - min_c
                
                # Check if the same movement from train1 is valid for train2
                if self.is_valid_move(actual_shape, train1['delta_row'], train1['delta_col'], self.size):
                    # Create train2 output grid using same movement as train1
                    train2_output_grid = self.move_shape(train2_input_grid, train2_center_row, train2_center_col, 
                                                       train1['delta_row'], train1['delta_col'])
                    
                    train2 = {
                        "input": train2_input_grid,
                        "output": train2_output_grid,
                        "center_row": train2_center_row,
                        "center_col": train2_center_col,
                        "delta_row": train1['delta_row'],  # Same as train1
                        "delta_col": train1['delta_col'],  # Same as train1
                        "value": train2_value
                    }
                    break
            
            # Generate test example with different shape/position but same movement
            while True:
                # Generate a random shape with center
                shape_template, template_center_r, template_center_c = self.generate_random_shape()
                
                # Get shape bounds to determine valid placement
                min_r, min_c, max_r, max_c = self.get_shape_bounds(shape_template)
                shape_height = max_r - min_r + 1
                shape_width = max_c - min_c + 1
                
                # Place shape at a random valid position
                max_start_row = self.size - shape_height
                max_start_col = self.size - shape_width
                
                if max_start_row <= 0 or max_start_col <= 0:
                    continue
                
                start_row = random.randint(0, max_start_row)
                start_col = random.randint(0, max_start_col)
                
                # Create test input grid with shape
                test_input_grid = self.create_empty_grid(self.size)
                test_value = random.randint(1, 9)
                
                actual_shape = []
                for r, c in shape_template:
                    actual_row = start_row + r - min_r
                    actual_col = start_col + c - min_c
                    test_input_grid[actual_row][actual_col] = test_value
                    actual_shape.append((actual_row, actual_col))
                
                # Calculate actual center coordinates
                test_center_row = start_row + template_center_r - min_r
                test_center_col = start_col + template_center_c - min_c
                
                # Check if the same movement from train is valid for test
                if self.is_valid_move(actual_shape, train1['delta_row'], train1['delta_col'], self.size):
                    # Create test output grid using same movement as train
                    test_output_grid = self.move_shape(test_input_grid, test_center_row, test_center_col, 
                                                     train1['delta_row'], train1['delta_col'])
                    
                    test = {
                        "input": test_input_grid,
                        "output": test_output_grid,
                        "center_row": test_center_row,
                        "center_col": test_center_col,
                        "delta_row": train1['delta_row'],  # Same as train
                        "delta_col": train1['delta_col'],  # Same as train
                        "value": test_value
                    }
                    break

            # In create_fewshot_examples method:
            solution = [
                f"shape_coords, value = MoveShapeGenerator.detect_shape_at_center(test_input, {test['center_row']}, {test['center_col']})",
                f"output_grid = MoveShapeGenerator.move_shape(test_input, {test['center_row']}, {test['center_col']}, {test['delta_row']}, {test['delta_col']})"
            ]

            examples.append({
                "train_input1": train1["input"],
                "train_output1": train1["output"],
                "train_input2": train2["input"],
                "train_output2": train2["output"],
                "test_input": test["input"],
                "test_output": test["output"],
                "solution": solution
            })

        return examples