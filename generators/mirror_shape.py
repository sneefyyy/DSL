# mirror_shape.py
import random
import copy
from .base import ExampleGenerator

class MirrorShapeGenerator(ExampleGenerator):
    """
    A class to generate few-shot examples for the 'mirror shape' task.
    """
    
    def __init__(self, size=5):
        super().__init__(size)
    
    @staticmethod
    def mirror_horizontal(grid):
        """Mirror the grid horizontally (left ↔ right)"""
        result = copy.deepcopy(grid)
        rows = len(result)
        cols = len(result[0]) if rows > 0 else 0
        
        for row in range(rows):
            for col in range(cols // 2):
                # Swap left and right
                result[row][col], result[row][cols - 1 - col] = \
                    result[row][cols - 1 - col], result[row][col]
        
        return result
    
    @staticmethod
    def mirror_vertical(grid):
        """Mirror the grid vertically (top ↔ bottom)"""
        result = copy.deepcopy(grid)
        rows = len(result)
        
        for row in range(rows // 2):
            # Swap top and bottom rows
            result[row], result[rows - 1 - row] = \
                result[rows - 1 - row], result[row]
        
        return result
    
    def create_fewshot_examples(self, num_examples=100):
        """Generate examples with consistent mirror patterns."""
        examples = []
        
        for i in range(num_examples):
            # Choose pattern: always mirror horizontally or vertically
            mirror_type = random.choice(["horizontal", "vertical"])
            
            examples_data = []
            for j in range(3):  # train1, train2, test
                # Create a grid with a shape on one side
                input_grid = self.create_empty_grid(self.size)
                
                # Add some random shapes
                color = random.randint(1, 9)
                if mirror_type == "horizontal":
                    # Put shape on left side only
                    for _ in range(random.randint(3, 6)):
                        row = random.randint(0, self.size - 1)
                        col = random.randint(0, self.size // 2 - 1)
                        input_grid[row][col] = color
                else:  # vertical
                    # Put shape on top half only
                    for _ in range(random.randint(3, 6)):
                        row = random.randint(0, self.size // 2 - 1)
                        col = random.randint(0, self.size - 1)
                        input_grid[row][col] = color
                
                # Apply mirror
                if mirror_type == "horizontal":
                    output_grid = self.mirror_horizontal(input_grid)
                else:
                    output_grid = self.mirror_vertical(input_grid)
                
                examples_data.append({
                    'input': input_grid,
                    'output': output_grid,
                    'mirror_type': mirror_type
                })
            
            train1, train2, test = examples_data
            
            # In create_fewshot_examples method:
            if mirror_type == "horizontal":
                solution = ["output_grid = MirrorShapeGenerator.mirror_horizontal(test_input)"]
            else:
                solution = ["output_grid = MirrorShapeGenerator.mirror_vertical(test_input)"]
            
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