import random
import copy
from .base import ExampleGenerator

class GravityGenerator(ExampleGenerator):
    """
    A class to generate few-shot examples for the 'apply gravity' task.
    Objects fall in a specified direction until blocked by another object or grid edge.
    """
    
    def __init__(self, size=8):
        """
        Initialize the GravityGenerator with a specified grid size.
        
        Args:
            size (int): The dimension of the square grid.
        """
        super().__init__(size)
    
    @staticmethod
    def apply_gravity(grid, direction):
        """
        Apply gravity to all objects in the grid, making them fall in the specified direction
        until blocked by another object or the grid edge.
        
        Args:
            grid (list): 2D grid with objects (non-zero values)
            direction (str): Direction of gravity - 'down', 'up', 'left', or 'right'
            
        Returns:
            list: New grid with gravity applied
        """
        if not grid or not grid[0]:
            return grid
        
        # Deep copy to avoid modifying original
        result = copy.deepcopy(grid)
        rows, cols = len(result), len(result[0])
        
        if direction == 'down':
            # Process columns from bottom to top
            for col in range(cols):
                # Collect all non-zero values in this column
                values = []
                for row in range(rows):
                    if result[row][col] != 0:
                        values.append(result[row][col])
                        result[row][col] = 0
                
                # Place values from bottom up
                for i, value in enumerate(values):
                    result[rows - 1 - i][col] = value
        
        elif direction == 'up':
            # Process columns from top to bottom
            for col in range(cols):
                # Collect all non-zero values in this column
                values = []
                for row in range(rows):
                    if result[row][col] != 0:
                        values.append(result[row][col])
                        result[row][col] = 0
                
                # Place values from top down
                for i, value in enumerate(values):
                    result[i][col] = value
        
        elif direction == 'left':
            # Process rows from left to right
            for row in range(rows):
                # Collect all non-zero values in this row
                values = []
                for col in range(cols):
                    if result[row][col] != 0:
                        values.append(result[row][col])
                        result[row][col] = 0
                
                # Place values from left to right
                for i, value in enumerate(values):
                    result[row][i] = value
        
        elif direction == 'right':
            # Process rows from right to left
            for row in range(rows):
                # Collect all non-zero values in this row
                values = []
                for col in range(cols):
                    if result[row][col] != 0:
                        values.append(result[row][col])
                        result[row][col] = 0
                
                # Place values from right to left
                for i, value in enumerate(values):
                    result[row][cols - 1 - i] = value
        
        return result
    
    def create_grid_with_objects(self, num_objects=None):
        """
        Create a grid with randomly placed objects.
        
        Args:
            num_objects (int): Number of objects to place. If None, uses random count.
            
        Returns:
            list: Grid with objects placed
        """
        grid = self.create_empty_grid(self.size)
        
        if num_objects is None:
            num_objects = random.randint(3, min(8, self.size * self.size // 3))
        
        # Place objects at random positions
        positions = set()
        attempts = 0
        while len(positions) < num_objects and attempts < num_objects * 10:
            row = random.randint(0, self.size - 1)
            col = random.randint(0, self.size - 1)
            if (row, col) not in positions:
                positions.add((row, col))
                grid[row][col] = random.randint(1, 9)
            attempts += 1
        
        return grid
    
    def generate_gravity_example(self, direction=None):
        """
        Generate a single gravity example.
        
        Args:
            direction (str): Direction of gravity. If None, chosen randomly.
            
        Returns:
            dict: Example with input, output, and parameters
        """
        if direction is None:
            direction = random.choice(['down', 'up', 'left', 'right'])
        
        # Create input grid with objects
        input_grid = self.create_grid_with_objects()
        
        # Apply gravity
        output_grid = self.apply_gravity(input_grid, direction)
        
        return {
            "input": input_grid,
            "output": output_grid,
            "direction": direction
        }
    
    def create_fewshot_examples(self, num_examples=100):
        """
        Generate multiple few-shot examples with consistent gravity direction.
        
        Args:
            num_examples (int): Number of examples to generate
            
        Returns:
            list: List of few-shot examples
        """
        examples = []
        
        for i in range(num_examples):
            # Pick a consistent direction for all three examples (train1, train2, test)
            direction = random.choice(['down', 'up', 'left', 'right'])
            
            # Generate training example 1
            train1 = self.generate_gravity_example(direction)
            
            # Generate training example 2 with same direction
            train2 = self.generate_gravity_example(direction)
            
            # Generate test example with same direction
            test = self.generate_gravity_example(direction)
            
            # Create solution
            solution = [
                f"output_grid = GravityGenerator.apply_gravity(test_input, '{direction}')"
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
