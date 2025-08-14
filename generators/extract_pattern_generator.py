# extract_pattern.py
import random
import copy
from .base import ExampleGenerator

class ExtractPatternGenerator(ExampleGenerator):
    """
    A class to generate few-shot examples for the 'extract pattern' task.
    """
    
    def __init__(self, size=5):
        """
        Initialize the ExtractPatternGenerator with a specified grid size.
        
        Args:
            size (int): The dimension of the square grid.
        """
        super().__init__(size)

    @staticmethod
    def extract_most_frequent_color(grid):
        """Extract all cells of the most frequently occurring color."""
        counts = {}
        for row in grid:
            for cell in row:
                if cell != 0:
                    counts[cell] = counts.get(cell, 0) + 1
        
        if counts:
            target_color = max(counts, key=counts.get)
            return ExtractPatternGenerator.extract_color(grid, target_color)
        return grid

    @staticmethod
    def extract_least_frequent_color(grid):
        """Extract all cells of the least frequently occurring color."""
        counts = {}
        for row in grid:
            for cell in row:
                if cell != 0:
                    counts[cell] = counts.get(cell, 0) + 1
        
        if counts:
            target_color = min(counts, key=counts.get)
            return ExtractPatternGenerator.extract_color(grid, target_color)
        return grid
    
    @staticmethod
    def extract_largest_shape(grid):
        """
        Extract the largest connected component (shape) from the grid.
        Uses 4-connectivity (up, down, left, right).
        
        Args:
            grid (list): Input grid
            
        Returns:
            list: New grid with only the largest shape
        """
        if not grid or not grid[0]:
            return grid
        
        rows, cols = len(grid), len(grid[0])
        visited = [[False] * cols for _ in range(rows)]
        
        def dfs(row, col, value):
            """Depth-first search to find connected component."""
            if (row < 0 or row >= rows or col < 0 or col >= cols or
                visited[row][col] or grid[row][col] != value or grid[row][col] == 0):
                return []
            
            visited[row][col] = True
            cells = [(row, col)]
            
            # Check 4 neighbors
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                cells.extend(dfs(row + dr, col + dc, value))
            
            return cells
        
        # Find all connected components
        components = []
        for row in range(rows):
            for col in range(cols):
                if not visited[row][col] and grid[row][col] != 0:
                    component = dfs(row, col, grid[row][col])
                    if component:
                        components.append((component, grid[row][col]))
        
        # Create output grid with only the largest component
        output = [[0] * cols for _ in range(rows)]
        
        if components:
            # Find largest component
            largest_component, value = max(components, key=lambda x: len(x[0]))
            
            # Fill in the largest component
            for row, col in largest_component:
                output[row][col] = value
        
        return output
    
    @staticmethod
    def extract_color(grid, color):
        """
        Extract all cells of a specific color/value.
        
        Args:
            grid (list): Input grid
            color (int): The value to extract
            
        Returns:
            list: New grid with only cells of the specified color
        """
        rows, cols = len(grid), len(grid[0])
        output = [[0] * cols for _ in range(rows)]
        
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == color:
                    output[row][col] = color
        
        return output
    
    @staticmethod
    def extract_smallest_shape(grid):
        """
        Extract the smallest connected component from the grid.
        
        Args:
            grid (list): Input grid
            
        Returns:
            list: New grid with only the smallest shape
        """
        if not grid or not grid[0]:
            return grid
        
        rows, cols = len(grid), len(grid[0])
        visited = [[False] * cols for _ in range(rows)]
        
        def dfs(row, col, value):
            """Depth-first search to find connected component."""
            if (row < 0 or row >= rows or col < 0 or col >= cols or
                visited[row][col] or grid[row][col] != value or grid[row][col] == 0):
                return []
            
            visited[row][col] = True
            cells = [(row, col)]
            
            # Check 4 neighbors
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                cells.extend(dfs(row + dr, col + dc, value))
            
            return cells
        
        # Find all connected components
        components = []
        for row in range(rows):
            for col in range(cols):
                if not visited[row][col] and grid[row][col] != 0:
                    component = dfs(row, col, grid[row][col])
                    if component:
                        components.append((component, grid[row][col]))
        
        # Create output grid with only the smallest component
        output = [[0] * cols for _ in range(rows)]
        
        if components:
            # Find smallest component
            smallest_component, value = min(components, key=lambda x: len(x[0]))
            
            # Fill in the smallest component
            for row, col in smallest_component:
                output[row][col] = value
        
        return output
    
    @staticmethod
    def extract_non_background(grid):
        """
        Extract all non-zero (non-background) cells.
        
        Args:
            grid (list): Input grid
            
        Returns:
            list: Grid with all non-zero cells preserved
        """
        return [[cell if cell != 0 else 0 for cell in row] for row in grid]
    
    def create_grid_with_shapes(self, num_shapes):
        """
        Create a grid with multiple distinct shapes of different sizes.
        
        Args:
            num_shapes (int): Number of shapes to create
            
        Returns:
            list: Grid with shapes
        """
        grid = self.create_empty_grid(self.size)
        used_positions = set()
        
        for i in range(num_shapes):
            color = random.randint(1, 9)
            shape_size = random.randint(1, 5)  # Variable shape sizes
            
            # Try to place shape
            attempts = 0
            while attempts < 50:
                start_row = random.randint(0, self.size - 1)
                start_col = random.randint(0, self.size - 1)
                
                # Create a connected shape
                shape_cells = [(start_row, start_col)]
                current_cells = [(start_row, start_col)]
                
                while len(shape_cells) < shape_size and current_cells:
                    # Pick a random cell to expand from
                    base = random.choice(current_cells)
                    
                    # Try to add adjacent cells
                    adjacent = []
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        new_row, new_col = base[0] + dr, base[1] + dc
                        if (0 <= new_row < self.size and 
                            0 <= new_col < self.size and
                            (new_row, new_col) not in shape_cells and
                            (new_row, new_col) not in used_positions):
                            adjacent.append((new_row, new_col))
                    
                    if adjacent:
                        new_cell = random.choice(adjacent)
                        shape_cells.append(new_cell)
                        current_cells.append(new_cell)
                    else:
                        current_cells.remove(base)
                
                # Check if we got a valid shape
                if len(shape_cells) >= 1:
                    # Place the shape
                    for row, col in shape_cells:
                        grid[row][col] = color
                        used_positions.add((row, col))
                    break
                
                attempts += 1
        
        return grid
    
    def generate_extraction_example(self, extraction_type):
        """
        Generate a single extraction example.
        
        Args:
            extraction_type (str): Type of extraction pattern
            
        Returns:
            dict: Example with input and output
        """
        if extraction_type == "largest_shape":
            # Create grid with 2-4 shapes of different sizes
            input_grid = self.create_grid_with_shapes(random.randint(2, 4))
            output_grid = self.extract_largest_shape(input_grid)
            
        elif extraction_type == "smallest_shape":
            # Create grid with 2-4 shapes of different sizes
            input_grid = self.create_grid_with_shapes(random.randint(2, 4))
            output_grid = self.extract_smallest_shape(input_grid)
            
        elif extraction_type == "specific_color":
            # Create grid with multiple colors
            input_grid = self.create_grid_with_shapes(random.randint(3, 5))
            
            # Find all colors in the grid
            colors = set()
            for row in input_grid:
                for cell in row:
                    if cell != 0:
                        colors.add(cell)
            
            if colors:
                # Pick a random color to extract
                target_color = random.choice(list(colors))
                output_grid = self.extract_color(input_grid, target_color)
            else:
                output_grid = input_grid
                target_color = 0
            
            return {
                "input": input_grid,
                "output": output_grid,
                "target_color": target_color
            }
            
        else:  # non_background
            # Create grid with some background
            input_grid = self.create_grid_with_shapes(random.randint(2, 3))
            output_grid = self.extract_non_background(input_grid)
        
        return {
            "input": input_grid,
            "output": output_grid
        }
    
    def create_fewshot_examples(self, num_examples=100):
        """
        Generate multiple few-shot examples with consistent extraction patterns.
        """
        examples = []
        
        # Include color extraction patterns with learnable rules
        extraction_types = ["largest_shape", "smallest_shape", "most_frequent_color", "least_frequent_color"]
        
        for i in range(num_examples):
            extraction_type = extraction_types[i % len(extraction_types)]
            
            if extraction_type == "most_frequent_color":
                # Generate examples
                train1 = self.generate_extraction_example("specific_color")
                train2 = self.generate_extraction_example("specific_color")
                test = self.generate_extraction_example("specific_color")
                
                # Apply the most frequent color rule to each
                for example in [train1, train2, test]:
                    counts = {}
                    for row in example["input"]:
                        for cell in row:
                            if cell != 0:
                                counts[cell] = counts.get(cell, 0) + 1
                    
                    if counts:
                        target_color = max(counts, key=counts.get)
                        example["output"] = self.extract_color(example["input"], target_color)
                
                solution = ["output_grid = ExtractPatternGenerator.extract_most_frequent_color(test_input)"]
                
            elif extraction_type == "least_frequent_color":
                # Generate examples
                train1 = self.generate_extraction_example("specific_color")
                train2 = self.generate_extraction_example("specific_color")
                test = self.generate_extraction_example("specific_color")
                
                # Apply the least frequent color rule to each
                for example in [train1, train2, test]:
                    counts = {}
                    for row in example["input"]:
                        for cell in row:
                            if cell != 0:
                                counts[cell] = counts.get(cell, 0) + 1
                    
                    if counts:
                        target_color = min(counts, key=counts.get)
                        example["output"] = self.extract_color(example["input"], target_color)
                
                solution = ["output_grid = ExtractPatternGenerator.extract_least_frequent_color(test_input)"]
                
            else:
                # Generate examples for shape-based extraction
                train1 = self.generate_extraction_example(extraction_type)
                train2 = self.generate_extraction_example(extraction_type)
                test = self.generate_extraction_example(extraction_type)
                
                # Create solution based on extraction type
                if extraction_type == "largest_shape":
                    solution = ["output_grid = ExtractPatternGenerator.extract_largest_shape(test_input)"]
                elif extraction_type == "smallest_shape":
                    solution = ["output_grid = ExtractPatternGenerator.extract_smallest_shape(test_input)"]
            
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