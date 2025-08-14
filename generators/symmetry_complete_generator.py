import random
import copy
from .base import ExampleGenerator

class SymmetryCompleteGenerator(ExampleGenerator):
    """
    A class to generate few-shot examples for the 'complete symmetry' task.
    """
    
    def __init__(self, size=5):
        super().__init__(size)
    
    @staticmethod
    def complete_horizontal_symmetry(grid):
        """
        Complete horizontal symmetry (mirror across vertical axis in center).
        Assumes left half is given, completes right half.
        """
        result = copy.deepcopy(grid)
        rows = len(result)
        cols = len(result[0]) if rows > 0 else 0
        
        # Mirror from left to right
        for row in range(rows):
            for col in range(cols // 2):
                # Copy left side to right side
                result[row][cols - 1 - col] = result[row][col]
        
        return result
    
    @staticmethod
    def complete_vertical_symmetry(grid):
        """
        Complete vertical symmetry (mirror across horizontal axis in center).
        Assumes top half is given, completes bottom half.
        """
        result = copy.deepcopy(grid)
        rows = len(result)
        cols = len(result[0]) if rows > 0 else 0
        
        # Mirror from top to bottom
        for row in range(rows // 2):
            for col in range(cols):
                # Copy top side to bottom side
                result[rows - 1 - row][col] = result[row][col]
        
        return result
    
    @staticmethod
    def complete_diagonal_symmetry(grid):
        """
        Complete diagonal symmetry (mirror across main diagonal).
        Assumes upper triangle is given, completes lower triangle.
        """
        result = copy.deepcopy(grid)
        size = min(len(result), len(result[0]) if result else 0)
        
        # Mirror across main diagonal
        for row in range(size):
            for col in range(row):
                # Copy upper triangle to lower triangle
                result[col][row] = result[row][col]
        
        return result
    
    @staticmethod
    def complete_rotational_symmetry(grid, order=2):
        """
        Complete rotational symmetry.
        For order=2 (180° symmetry), assumes top-left quadrant is given.
        """
        result = copy.deepcopy(grid)
        rows = len(result)
        cols = len(result[0]) if rows > 0 else 0
        
        if order == 2:  # 180° rotational symmetry
            for row in range(rows):
                for col in range(cols):
                    # If current cell is non-zero, copy to rotated position
                    if result[row][col] != 0:
                        # 180° rotation: (row, col) -> (rows-1-row, cols-1-col)
                        result[rows - 1 - row][cols - 1 - col] = result[row][col]
        
        return result
    
    def create_partial_symmetric_shape(self, symmetry_type="horizontal"):
        """
        Create a grid with partial symmetric content.
        
        Args:
            symmetry_type (str): Type of symmetry to create partial content for
            
        Returns:
            tuple: (input_grid, expected_output_grid)
        """
        input_grid = self.create_empty_grid(self.size)
        
        if symmetry_type == "horizontal":
            # Create content only on left half
            color = random.randint(1, 9)
            
            # Create some pattern on left side
            patterns = ["diagonal", "random", "L_shape", "square"]
            pattern = random.choice(patterns)
            
            if pattern == "diagonal":
                for i in range(self.size // 2):
                    if i < self.size:
                        input_grid[i][i] = color
            
            elif pattern == "L_shape":
                # Vertical line
                for i in range(3):
                    input_grid[i][0] = color
                # Horizontal line
                for j in range(self.size // 2):
                    input_grid[2][j] = color
            
            elif pattern == "square":
                for i in range(2):
                    for j in range(2):
                        input_grid[i][j] = color
            
            else:  # random
                for _ in range(random.randint(3, 6)):
                    row = random.randint(0, self.size - 1)
                    col = random.randint(0, self.size // 2 - 1)
                    input_grid[row][col] = color
            
            output_grid = self.complete_horizontal_symmetry(input_grid)
        
        elif symmetry_type == "vertical":
            # Create content only on top half
            color = random.randint(1, 9)
            
            patterns = ["horizontal_line", "random", "triangle"]
            pattern = random.choice(patterns)
            
            if pattern == "horizontal_line":
                row = random.randint(0, self.size // 2 - 1)
                for col in range(self.size):
                    input_grid[row][col] = color
            
            elif pattern == "triangle":
                for i in range(min(3, self.size // 2)):
                    for j in range(i + 1):
                        if j < self.size:
                            input_grid[i][j] = color
            
            else:  # random
                for _ in range(random.randint(3, 6)):
                    row = random.randint(0, self.size // 2 - 1)
                    col = random.randint(0, self.size - 1)
                    input_grid[row][col] = color
            
            output_grid = self.complete_vertical_symmetry(input_grid)
        
        elif symmetry_type == "diagonal":
            # Create content only in upper triangle
            color = random.randint(1, 9)
            
            # Add some cells in upper triangle
            for _ in range(random.randint(3, 6)):
                row = random.randint(0, self.size - 2)
                col = random.randint(row + 1, self.size - 1)
                input_grid[row][col] = color
            
            output_grid = self.complete_diagonal_symmetry(input_grid)
        
        else:  # rotational
            # Create content in one quadrant
            color = random.randint(1, 9)
            
            # Add pattern in top-left quadrant
            for _ in range(random.randint(2, 4)):
                row = random.randint(0, self.size // 2 - 1)
                col = random.randint(0, self.size // 2 - 1)
                input_grid[row][col] = color
            
            output_grid = self.complete_rotational_symmetry(input_grid)
        
        return input_grid, output_grid
    
    def create_fewshot_examples(self, num_examples=100):
        """
        Generate multiple few-shot examples with consistent symmetry patterns.
        
        Args:
            num_examples (int): Number of examples to generate
            
        Returns:
            list: List of few-shot examples
        """
        examples = []
        
        # Symmetry types to cycle through
        symmetry_types = ["horizontal", "vertical", "diagonal", "rotational"]
        
        for i in range(num_examples):
            # Choose a symmetry type for this example set
            symmetry_type = symmetry_types[i % len(symmetry_types)]
            
            examples_data = []
            
            for j in range(3):  # train1, train2, test
                input_grid, output_grid = self.create_partial_symmetric_shape(symmetry_type)
                
                examples_data.append({
                    'input': input_grid,
                    'output': output_grid,
                    'symmetry_type': symmetry_type
                })
            
            train1, train2, test = examples_data
            
            # In create_fewshot_examples method:
            if symmetry_type == "horizontal":
                solution = ["output_grid = SymmetryCompleteGenerator.complete_horizontal_symmetry(test_input)"]
            elif symmetry_type == "vertical":
                solution = ["output_grid = SymmetryCompleteGenerator.complete_vertical_symmetry(test_input)"]
            elif symmetry_type == "diagonal":
                solution = ["output_grid = SymmetryCompleteGenerator.complete_diagonal_symmetry(test_input)"]
            else:  # rotational
                solution = ["output_grid = SymmetryCompleteGenerator.complete_rotational_symmetry(test_input)"]
            
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
    