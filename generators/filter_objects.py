import random
import copy
from collections import Counter
from .base import ExampleGenerator

class FilterObjectsGenerator(ExampleGenerator):
    """
    A class to generate few-shot examples for the 'filter objects' task.
    Handles filtering objects based on criteria like color lists, frequency, or position.
    """
    
    def __init__(self, size=5):
        """
        Initialize the FilterObjectsGenerator with a specified grid size.
        
        Args:
            size (int): The dimension of the square grid.
        """
        super().__init__(size)
    
    @staticmethod
    def filter_objects(grid, criteria):
        """
        Filter objects in the grid based on the specified criteria.
        
        Args:
            grid (list): 2D grid with objects (non-zero values)
            criteria (str): Filtering criteria:
                - "keep_colors=[3,5,9]": Keep only specified colors
                - "remove_colors=[6,8]": Remove specified colors
                - "keep_minority": Keep only the least frequent color
                - "keep_majority": Keep only the most frequent color
                - "keep_column=2": Keep only colors that appear in specified column
                - "keep_row=1": Keep only colors that appear in specified row
            
        Returns:
            list: New grid with filtering applied
        """
        if not grid or not grid[0]:
            return grid
        
        # Deep copy to avoid modifying original
        result = copy.deepcopy(grid)
        rows, cols = len(result), len(result[0])
        
        if criteria == "keep_minority":
            # Count non-zero color frequencies
            counts = Counter(cell for row in grid for cell in row if cell != 0)
            if not counts:
                return result
            minority_color = min(counts, key=counts.get)
            
            # Remove all colors except minority
            for i in range(rows):
                for j in range(cols):
                    if result[i][j] != minority_color and result[i][j] != 0:
                        result[i][j] = 0
        
        elif criteria == "keep_majority":
            # Count non-zero color frequencies
            counts = Counter(cell for row in grid for cell in row if cell != 0)
            if not counts:
                return result
            majority_color = max(counts, key=counts.get)
            
            # Remove all colors except majority
            for i in range(rows):
                for j in range(cols):
                    if result[i][j] != majority_color and result[i][j] != 0:
                        result[i][j] = 0
        
        elif criteria.startswith("keep_colors="):
            # Parse color list: "keep_colors=[3,5,9]"
            color_str = criteria.split("=", 1)[1]
            keep_list = eval(color_str)
            
            # Remove all colors not in keep list
            for i in range(rows):
                for j in range(cols):
                    if result[i][j] not in keep_list and result[i][j] != 0:
                        result[i][j] = 0
        
        elif criteria.startswith("remove_colors="):
            # Parse color list: "remove_colors=[6,8]"
            color_str = criteria.split("=", 1)[1]
            remove_list = eval(color_str)
            
            # Remove specified colors
            for i in range(rows):
                for j in range(cols):
                    if result[i][j] in remove_list:
                        result[i][j] = 0
        
        elif criteria.startswith("keep_column="):
            # Parse column index: "keep_column=2"
            col_idx = int(criteria.split("=")[1])
            
            # Get colors in specified column
            if 0 <= col_idx < cols:
                column_colors = set(grid[i][col_idx] for i in range(rows) if grid[i][col_idx] != 0)
                
                # Remove all colors not in that column
                for i in range(rows):
                    for j in range(cols):
                        if result[i][j] not in column_colors and result[i][j] != 0:
                            result[i][j] = 0
        
        elif criteria.startswith("keep_row="):
            # Parse row index: "keep_row=1"
            row_idx = int(criteria.split("=")[1])
            
            # Get colors in specified row
            if 0 <= row_idx < rows:
                row_colors = set(grid[row_idx][j] for j in range(cols) if grid[row_idx][j] != 0)
                
                # Remove all colors not in that row
                for i in range(rows):
                    for j in range(cols):
                        if result[i][j] not in row_colors and result[i][j] != 0:
                            result[i][j] = 0
        
        return result
    
    def create_grid_with_mixed_colors(self, num_colors=3, density=0.5):
        """
        Create a grid with multiple colors at random positions.
        
        Args:
            num_colors (int): Number of different colors to use
            density (float): Approximate fraction of cells to fill (0.0 to 1.0)
            
        Returns:
            tuple: (grid, colors_used)
        """
        grid = self.create_empty_grid(self.size)
        colors = random.sample(range(1, 10), num_colors)
        
        # Fill grid with random colors
        for i in range(self.size):
            for j in range(self.size):
                if random.random() < density:
                    grid[i][j] = random.choice(colors)
        
        return grid, colors
    
    def create_grid_with_frequency_pattern(self, minority_count=None, majority_count=None):
        """
        Create a grid with colors having specific frequency distributions.
        
        Args:
            minority_count (int): Target count for minority color
            majority_count (int): Target count for majority color
            
        Returns:
            tuple: (grid, minority_color, majority_color)
        """
        grid = self.create_empty_grid(self.size)
        
        # Choose two colors
        color1, color2 = random.sample(range(1, 10), 2)
        
        # Set counts
        if minority_count is None:
            minority_count = random.randint(3, 8)
        if majority_count is None:
            majority_count = random.randint(minority_count + 3, min(15, self.size * self.size - 5))
        
        # Determine which is minority
        minority_color = color1
        majority_color = color2
        min_count = minority_count
        maj_count = majority_count
        
        # Collect all positions
        all_positions = [(i, j) for i in range(self.size) for j in range(self.size)]
        random.shuffle(all_positions)
        
        # Place minority color
        for i in range(min(min_count, len(all_positions))):
            row, col = all_positions[i]
            grid[row][col] = minority_color
        
        # Place majority color
        for i in range(min_count, min(min_count + maj_count, len(all_positions))):
            row, col = all_positions[i]
            grid[row][col] = majority_color
        
        return grid, minority_color, majority_color
    
    def generate_filter_objects_example(self, criteria_type=None):
        """
        Generate a single filter_objects example.
        
        Args:
            criteria_type (str): Type of criteria. If None, chosen randomly.
            
        Returns:
            dict: Example with input, output, and parameters
        """
        if criteria_type is None:
            criteria_type = random.choice([
                "keep_colors",
                "remove_colors",
                "keep_minority",
                "keep_majority",
                "keep_column",
                "keep_row"
            ])
        
        if criteria_type == "keep_colors":
            # Create grid with multiple colors
            num_colors = random.randint(3, 5)
            input_grid, all_colors = self.create_grid_with_mixed_colors(num_colors, density=0.6)
            
            # Choose subset to keep
            num_keep = random.randint(1, len(all_colors) - 1)
            keep_colors = random.sample(all_colors, num_keep)
            criteria = f"keep_colors={keep_colors}"
            
        elif criteria_type == "remove_colors":
            # Create grid with multiple colors
            num_colors = random.randint(3, 5)
            input_grid, all_colors = self.create_grid_with_mixed_colors(num_colors, density=0.6)
            
            # Choose subset to remove
            num_remove = random.randint(1, len(all_colors) - 1)
            remove_colors = random.sample(all_colors, num_remove)
            criteria = f"remove_colors={remove_colors}"
            
        elif criteria_type == "keep_minority":
            input_grid, minority_color, majority_color = self.create_grid_with_frequency_pattern()
            criteria = "keep_minority"
            
        elif criteria_type == "keep_majority":
            input_grid, minority_color, majority_color = self.create_grid_with_frequency_pattern()
            criteria = "keep_majority"
            
        elif criteria_type == "keep_column":
            # Create grid and pick a column
            num_colors = random.randint(3, 5)
            input_grid, all_colors = self.create_grid_with_mixed_colors(num_colors, density=0.6)
            col_idx = random.randint(0, self.size - 1)
            criteria = f"keep_column={col_idx}"
            
        elif criteria_type == "keep_row":
            # Create grid and pick a row
            num_colors = random.randint(3, 5)
            input_grid, all_colors = self.create_grid_with_mixed_colors(num_colors, density=0.6)
            row_idx = random.randint(0, self.size - 1)
            criteria = f"keep_row={row_idx}"
        
        # Apply filtering
        output_grid = self.filter_objects(input_grid, criteria)
        
        return {
            "input": input_grid,
            "output": output_grid,
            "criteria": criteria
        }
    
    def create_fewshot_examples(self, num_examples=100):
        """
        Generate multiple few-shot examples with consistent filtering criteria.
        
        Args:
            num_examples (int): Number of examples to generate
            
        Returns:
            list: List of few-shot examples
        """
        examples = []
        
        for i in range(num_examples):
            # Pick a consistent criteria type for all three examples
            criteria_type = random.choice([
                "keep_colors",
                "remove_colors",
                "keep_minority",
                "keep_majority",
                "keep_column",
                "keep_row"
            ])
            
            # For criteria that need specific values, use same structure across all 3 examples
            if criteria_type in ["keep_minority", "keep_majority"]:
                # Generate three examples with same criteria
                train1 = self.generate_filter_objects_example(criteria_type)
                train2 = self.generate_filter_objects_example(criteria_type)
                test = self.generate_filter_objects_example(criteria_type)
                
            elif criteria_type in ["keep_colors", "remove_colors"]:
                # Use consistent color lists across examples
                num_colors = random.randint(3, 5)
                all_colors = random.sample(range(1, 10), num_colors)
                
                if criteria_type == "keep_colors":
                    num_keep = random.randint(1, num_colors - 1)
                    target_colors = random.sample(all_colors, num_keep)
                    criteria = f"keep_colors={target_colors}"
                else:  # remove_colors
                    num_remove = random.randint(1, num_colors - 1)
                    target_colors = random.sample(all_colors, num_remove)
                    criteria = f"remove_colors={target_colors}"
                
                # Generate grids with same color palette
                input1, _ = self.create_grid_with_mixed_colors(num_colors, density=0.6)
                # Replace colors with our chosen palette
                color_map = {i+1: all_colors[i % len(all_colors)] for i in range(9)}
                input1 = [[color_map.get(cell, cell) if cell != 0 else 0 for cell in row] for row in input1]
                output1 = self.filter_objects(input1, criteria)
                
                input2, _ = self.create_grid_with_mixed_colors(num_colors, density=0.6)
                input2 = [[color_map.get(cell, cell) if cell != 0 else 0 for cell in row] for row in input2]
                output2 = self.filter_objects(input2, criteria)
                
                input_test, _ = self.create_grid_with_mixed_colors(num_colors, density=0.6)
                input_test = [[color_map.get(cell, cell) if cell != 0 else 0 for cell in row] for row in input_test]
                output_test = self.filter_objects(input_test, criteria)
                
                train1 = {"input": input1, "output": output1, "criteria": criteria}
                train2 = {"input": input2, "output": output2, "criteria": criteria}
                test = {"input": input_test, "output": output_test, "criteria": criteria}
                
            else:  # keep_column or keep_row
                # Use same row/column index
                if criteria_type == "keep_column":
                    idx = random.randint(0, self.size - 1)
                    criteria = f"keep_column={idx}"
                else:
                    idx = random.randint(0, self.size - 1)
                    criteria = f"keep_row={idx}"
                
                # Generate three different grids with same criteria
                input1, _ = self.create_grid_with_mixed_colors(random.randint(3, 5), density=0.6)
                output1 = self.filter_objects(input1, criteria)
                
                input2, _ = self.create_grid_with_mixed_colors(random.randint(3, 5), density=0.6)
                output2 = self.filter_objects(input2, criteria)
                
                input_test, _ = self.create_grid_with_mixed_colors(random.randint(3, 5), density=0.6)
                output_test = self.filter_objects(input_test, criteria)
                
                train1 = {"input": input1, "output": output1, "criteria": criteria}
                train2 = {"input": input2, "output": output2, "criteria": criteria}
                test = {"input": input_test, "output": output_test, "criteria": criteria}
            
            # Create solution
            solution = [
                f"output_grid = FilterObjectsGenerator.filter_objects(test_input, '{test['criteria']}')"
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
