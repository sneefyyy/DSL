# repeat_pattern.py
import random
import copy
from .base import ExampleGenerator

class RepeatPatternGenerator(ExampleGenerator):
    """
    A class to generate few-shot examples for the 'repeat pattern' task.
    """
    
    def __init__(self, size=5):
        """
        Initialize the RepeatPatternGenerator with a specified grid size.
        
        Args:
            size (int): The dimension of the square grid.
        """
        super().__init__(size)
    
    @staticmethod
    def repeat_horizontal(grid, times):
        """
        Repeat the leftmost single column pattern horizontally.
        """
        if not grid or not grid[0]:
            return grid
        
        rows = len(grid)
        cols = len(grid[0])
        
        # Find the leftmost non-empty column
        pattern_col = None
        for col in range(cols):
            if any(grid[row][col] != 0 for row in range(rows)):
                pattern_col = col
                break
        
        if pattern_col is None:
            return grid
        
        # Create new grid
        new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
        
        # Repeat the column pattern
        for rep in range(times):
            target_col = pattern_col + rep
            if target_col < cols:
                for row in range(rows):
                    new_grid[row][target_col] = grid[row][pattern_col]
        
        return new_grid
    
    @staticmethod
    def repeat_vertical(grid, times):
        """
        Repeat the topmost single row pattern vertically.
        """
        if not grid or not grid[0]:
            return grid
        
        rows = len(grid)
        cols = len(grid[0])
        
        # Find the topmost non-empty row
        pattern_row = None
        for row in range(rows):
            if any(grid[row][col] != 0 for col in range(cols)):
                pattern_row = row
                break
        
        if pattern_row is None:
            return grid
        
        # Create new grid
        new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
        
        # Repeat the row pattern
        for rep in range(times):
            target_row = pattern_row + rep
            if target_row < rows:
                for col in range(cols):
                    new_grid[target_row][col] = grid[pattern_row][col]
        
        return new_grid
    
    @staticmethod
    def tile_pattern(grid):
        """
        Tile a single cell pattern to fill the entire grid.
        """
        if not grid or not grid[0]:
            return grid
        
        rows = len(grid)
        cols = len(grid[0])
        
        # Find the first non-zero cell
        pattern_value = 0
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] != 0:
                    pattern_value = grid[row][col]
                    break
            if pattern_value != 0:
                break
        
        if pattern_value == 0:
            return grid
        
        # Create new grid filled with the pattern
        new_grid = [[pattern_value for _ in range(cols)] for _ in range(rows)]
        
        return new_grid
    
    @staticmethod
    def repeat_diagonal(grid, times):
        """
        Repeat the pattern diagonally.
        """
        if not grid or not grid[0]:
            return grid
        
        rows = len(grid)
        cols = len(grid[0])
        
        # Find the first non-zero cell
        pattern_value = 0
        start_row, start_col = 0, 0
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] != 0:
                    pattern_value = grid[row][col]
                    start_row, start_col = row, col
                    break
            if pattern_value != 0:
                break
        
        if pattern_value == 0:
            return grid
        
        # Create new grid
        new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
        
        # Place pattern diagonally
        for rep in range(times):
            new_row = start_row + rep
            new_col = start_col + rep
            if new_row < rows and new_col < cols:
                new_grid[new_row][new_col] = pattern_value
        
        return new_grid
    
    def create_simple_pattern(self, pattern_type):
        """
        Create a simple pattern in a grid.
        
        Args:
            pattern_type (str): Type of pattern to create
            
        Returns:
            list: Grid with pattern
        """
        grid = self.create_empty_grid(self.size)
        color = random.randint(1, 9)
        
        if pattern_type == "single_column":
            # Single column pattern
            col = 0  # Always leftmost
            num_cells = random.randint(1, self.size)
            for row in range(num_cells):
                grid[row][col] = color
                
        elif pattern_type == "single_row":
            # Single row pattern
            row = 0  # Always topmost
            num_cells = random.randint(1, self.size)
            for col in range(num_cells):
                grid[row][col] = color
                
        elif pattern_type == "single_cell":
            # Single cell for tiling
            row = random.randint(0, 1)
            col = random.randint(0, 1)
            grid[row][col] = color
            
        elif pattern_type == "diagonal_start":
            # Single cell for diagonal pattern
            grid[0][0] = color
            
        return grid
    
    def generate_repeat_example(self, repeat_type, times=None):
        """
        Generate a single repeat example with consistent patterns.
        
        Args:
            repeat_type (str): Type of repetition
            times (int): Number of repetitions (if applicable)
            
        Returns:
            dict: Example with input and output
        """
        if repeat_type == "horizontal":
            pattern_type = "single_column"
            input_grid = self.create_simple_pattern(pattern_type)
            if times is None:
                times = random.randint(2, 4)
            output_grid = self.repeat_horizontal(input_grid, times)
            
        elif repeat_type == "vertical":
            pattern_type = "single_row"
            input_grid = self.create_simple_pattern(pattern_type)
            if times is None:
                times = random.randint(2, 4)
            output_grid = self.repeat_vertical(input_grid, times)
            
        elif repeat_type == "tile":
            pattern_type = "single_cell"
            input_grid = self.create_simple_pattern(pattern_type)
            output_grid = self.tile_pattern(input_grid)
            
        elif repeat_type == "diagonal":
            pattern_type = "diagonal_start"
            input_grid = self.create_simple_pattern(pattern_type)
            if times is None:
                times = random.randint(3, 5)
            output_grid = self.repeat_diagonal(input_grid, times)
        
        return {
            "input": input_grid,
            "output": output_grid,
            "times": times if times else None
        }
    
    def create_fewshot_examples(self, num_examples=100):
        """
        Generate multiple few-shot examples with consistent repetition patterns.
        
        Args:
            num_examples (int): Number of examples to generate
            
        Returns:
            list: List of few-shot examples
        """
        examples = []
        
        # Pattern types to cycle through
        repeat_types = ["horizontal", "vertical", "tile", "diagonal"]
        
        for i in range(num_examples):
            repeat_type = repeat_types[i % len(repeat_types)]
            
            if repeat_type == "horizontal":
                # Consistent number of horizontal repetitions
                times = random.randint(2, 5)
                
                train1 = self.generate_repeat_example(repeat_type, times)
                train2 = self.generate_repeat_example(repeat_type, times)
                test = self.generate_repeat_example(repeat_type, times)
                
                solution = [f"output_grid = RepeatPatternGenerator.repeat_horizontal(test_input, {times})"]
                
            elif repeat_type == "vertical":
                # Consistent number of vertical repetitions
                times = random.randint(2, 5)
                
                train1 = self.generate_repeat_example(repeat_type, times)
                train2 = self.generate_repeat_example(repeat_type, times)
                test = self.generate_repeat_example(repeat_type, times)
                
                solution = [f"output_grid = RepeatPatternGenerator.repeat_vertical(test_input, {times})"]
                
            elif repeat_type == "diagonal":
                # Consistent number of diagonal repetitions
                times = random.randint(3, 5)
                
                train1 = self.generate_repeat_example(repeat_type, times)
                train2 = self.generate_repeat_example(repeat_type, times)
                test = self.generate_repeat_example(repeat_type, times)
                
                solution = [f"output_grid = RepeatPatternGenerator.repeat_diagonal(test_input, {times})"]
                
            else:  # tile
                train1 = self.generate_repeat_example(repeat_type)
                train2 = self.generate_repeat_example(repeat_type)
                test = self.generate_repeat_example(repeat_type)
                
                solution = ["output_grid = RepeatPatternGenerator.tile_pattern(test_input)"]
            
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
    