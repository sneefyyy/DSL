import random
import copy
from .base import ExampleGenerator

class ModifyNeighborsGenerator(ExampleGenerator):
    """
    A class to generate few-shot examples for the 'modify neighbors' task.
    Handles operations like filling between pairs and marking adjacent cells.
    """
    
    def __init__(self, size=8):
        """
        Initialize the ModifyNeighborsGenerator with a specified grid size.
        
        Args:
            size (int): The dimension of the square grid.
        """
        super().__init__(size)
    
    @staticmethod
    def modify_neighbors(grid, target_color=1, operation="fill_between_pairs_horizontal"):
        """
        Modify neighbors of target-colored cells based on the specified operation.
        
        Args:
            grid (list): 2D grid with objects (non-zero values)
            target_color (int): The color to search for
            operation (str): Type of neighbor modification:
                - "fill_between_pairs_horizontal": Fill cells between pairs on same row
                - "fill_between_pairs_vertical": Fill cells between pairs on same column
                - "mark_adjacent_4": Mark 4-connected neighbors
                - "mark_diagonal_8": Mark 8-connected neighbors
            
        Returns:
            list: New grid with neighbors modified
        """
        if not grid or not grid[0]:
            return grid
        
        # Deep copy to avoid modifying original
        result = copy.deepcopy(grid)
        rows, cols = len(result), len(result[0])
        
        # Find the next available color for marking (avoid conflicts)
        used_colors = set()
        for row in result:
            for cell in row:
                if cell != 0:
                    used_colors.add(cell)
        new_color = 2  # Default
        for c in range(1, 10):
            if c not in used_colors and c != target_color:
                new_color = c
                break
        
        if operation == "fill_between_pairs_horizontal":
            # Fill cells between pairs of target_color on the same row
            for row_idx in range(rows):
                # Find all positions with target_color in this row
                positions = [col_idx for col_idx in range(cols) 
                           if result[row_idx][col_idx] == target_color]
                
                # Fill between consecutive pairs
                for i in range(len(positions) - 1):
                    start = positions[i]
                    end = positions[i + 1]
                    # Fill cells between (exclusive of endpoints)
                    for col_idx in range(start + 1, end):
                        if result[row_idx][col_idx] == 0:
                            result[row_idx][col_idx] = new_color
        
        elif operation == "fill_between_pairs_vertical":
            # Fill cells between pairs of target_color on the same column
            for col_idx in range(cols):
                # Find all positions with target_color in this column
                positions = [row_idx for row_idx in range(rows) 
                           if result[row_idx][col_idx] == target_color]
                
                # Fill between consecutive pairs
                for i in range(len(positions) - 1):
                    start = positions[i]
                    end = positions[i + 1]
                    # Fill cells between (exclusive of endpoints)
                    for row_idx in range(start + 1, end):
                        if result[row_idx][col_idx] == 0:
                            result[row_idx][col_idx] = new_color
        
        elif operation == "mark_adjacent_4":
            # Mark 4-connected neighbors of all target_color cells
            # First, find all target cells
            target_cells = []
            for row_idx in range(rows):
                for col_idx in range(cols):
                    if result[row_idx][col_idx] == target_color:
                        target_cells.append((row_idx, col_idx))
            
            # Mark neighbors
            for row_idx, col_idx in target_cells:
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    new_row = row_idx + dr
                    new_col = col_idx + dc
                    if (0 <= new_row < rows and 0 <= new_col < cols and 
                        result[new_row][new_col] == 0):
                        result[new_row][new_col] = new_color
        
        elif operation == "mark_diagonal_8":
            # Mark 8-connected neighbors of all target_color cells
            # First, find all target cells
            target_cells = []
            for row_idx in range(rows):
                for col_idx in range(cols):
                    if result[row_idx][col_idx] == target_color:
                        target_cells.append((row_idx, col_idx))
            
            # Mark all 8 neighbors
            for row_idx, col_idx in target_cells:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        new_row = row_idx + dr
                        new_col = col_idx + dc
                        if (0 <= new_row < rows and 0 <= new_col < cols and 
                            result[new_row][new_col] == 0):
                            result[new_row][new_col] = new_color
        
        return result
    
    def create_grid_with_pairs(self, operation):
        """
        Create a grid with pairs of objects arranged for the given operation.
        
        Args:
            operation (str): Type of operation to create examples for
            
        Returns:
            tuple: (grid, target_color)
        """
        grid = self.create_empty_grid(self.size)
        target_color = random.randint(1, 9)
        
        if operation == "fill_between_pairs_horizontal":
            # Place pairs on random rows
            num_rows_with_pairs = random.randint(1, min(3, self.size))
            selected_rows = random.sample(range(self.size), num_rows_with_pairs)
            
            for row_idx in selected_rows:
                # Place a pair with some space between
                space = random.randint(1, max(1, self.size - 3))
                start = random.randint(0, max(0, self.size - space - 2))
                end = start + space + 1
                if end < self.size:
                    grid[row_idx][start] = target_color
                    grid[row_idx][end] = target_color
        
        elif operation == "fill_between_pairs_vertical":
            # Place pairs on random columns
            num_cols_with_pairs = random.randint(1, min(3, self.size))
            selected_cols = random.sample(range(self.size), num_cols_with_pairs)
            
            for col_idx in selected_cols:
                # Place a pair with some space between
                space = random.randint(1, max(1, self.size - 3))
                start = random.randint(0, max(0, self.size - space - 2))
                end = start + space + 1
                if end < self.size:
                    grid[start][col_idx] = target_color
                    grid[end][col_idx] = target_color
        
        elif operation in ["mark_adjacent_4", "mark_diagonal_8"]:
            # Place random objects for neighbor marking
            num_objects = random.randint(2, min(5, self.size))
            positions = set()
            attempts = 0
            
            while len(positions) < num_objects and attempts < num_objects * 20:
                row = random.randint(1, self.size - 2)  # Not on edges for clearer examples
                col = random.randint(1, self.size - 2)
                if (row, col) not in positions:
                    positions.add((row, col))
                    grid[row][col] = target_color
                attempts += 1
        
        return grid, target_color
    
    def generate_modify_neighbors_example(self, operation=None):
        """
        Generate a single modify_neighbors example.
        
        Args:
            operation (str): Type of operation. If None, chosen randomly.
            
        Returns:
            dict: Example with input, output, and parameters
        """
        if operation is None:
            operation = random.choice([
                "fill_between_pairs_horizontal",
                "fill_between_pairs_vertical",
                "mark_adjacent_4",
                "mark_diagonal_8"
            ])
        
        # Create input grid with appropriate pattern
        input_grid, target_color = self.create_grid_with_pairs(operation)
        
        # Apply modify_neighbors
        output_grid = self.modify_neighbors(input_grid, target_color, operation)
        
        return {
            "input": input_grid,
            "output": output_grid,
            "target_color": target_color,
            "operation": operation
        }
    
    def create_fewshot_examples(self, num_examples=100):
        """
        Generate multiple few-shot examples with consistent operation.
        
        Args:
            num_examples (int): Number of examples to generate
            
        Returns:
            list: List of few-shot examples
        """
        examples = []
        
        for i in range(num_examples):
            # Pick a consistent operation for all three examples (train1, train2, test)
            operation = random.choice([
                "fill_between_pairs_horizontal",
                "fill_between_pairs_vertical",
                "mark_adjacent_4",
                "mark_diagonal_8"
            ])
            
            # Pick a consistent target color for all three examples
            target_color = random.randint(1, 9)
            
            # Generate training example 1
            input_grid1, _ = self.create_grid_with_pairs(operation)
            output_grid1 = self.modify_neighbors(input_grid1, target_color, operation)
            train1 = {
                "input": input_grid1,
                "output": output_grid1,
                "target_color": target_color,
                "operation": operation
            }
            
            # Generate training example 2 with same operation and target color
            input_grid2, _ = self.create_grid_with_pairs(operation)
            output_grid2 = self.modify_neighbors(input_grid2, target_color, operation)
            train2 = {
                "input": input_grid2,
                "output": output_grid2,
                "target_color": target_color,
                "operation": operation
            }
            
            # Generate test example with same operation and target color
            input_grid_test, _ = self.create_grid_with_pairs(operation)
            output_grid_test = self.modify_neighbors(input_grid_test, target_color, operation)
            test = {
                "input": input_grid_test,
                "output": output_grid_test,
                "target_color": target_color,
                "operation": operation
            }
            
            # Create solution
            solution = [
                f"output_grid = ModifyNeighborsGenerator.modify_neighbors(test_input, {target_color}, '{operation}')"
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
