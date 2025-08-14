import random
import copy
from .base import ExampleGenerator

class FloodFillGenerator(ExampleGenerator):
    """
    A class to generate few-shot examples for the 'flood fill' task.
    """
    
    def __init__(self, size=5):
        """
        Initialize the FloodFillGenerator with a specified grid size.
        
        Args:
            size (int): The dimension of the square grid.
        """
        super().__init__(size)
    
    @staticmethod
    def flood_fill(grid, start_point, value):
        """
        Perform flood fill starting from a point, replacing all connected cells
        of the same color with a new value.
        
        Args:
            grid (list): 2D grid to fill
            start_point (tuple): (row, col) starting point for flood fill
            value (int): New value to fill with
            
        Returns:
            list: New grid with flood fill applied
        """
        if not grid or not grid[0]:
            return grid
        
        # Deep copy to avoid modifying original
        result = copy.deepcopy(grid)
        rows, cols = len(result), len(result[0])
        start_row, start_col = start_point
        
        # Check bounds
        if not (0 <= start_row < rows and 0 <= start_col < cols):
            return result
        
        # Get the original color to replace
        original_color = result[start_row][start_col]
        
        # If the start point already has the target value, nothing to do
        if original_color == value:
            return result
        
        # Use a stack for iterative flood fill (to avoid recursion limits)
        stack = [(start_row, start_col)]
        
        while stack:
            row, col = stack.pop()
            
            # Check bounds
            if not (0 <= row < rows and 0 <= col < cols):
                continue
            
            # Check if this cell has the original color
            if result[row][col] != original_color:
                continue
            
            # Fill this cell
            result[row][col] = value
            
            # Add neighbors to stack (4-connected)
            stack.extend([
                (row - 1, col),  # up
                (row + 1, col),  # down
                (row, col - 1),  # left
                (row, col + 1)   # right
            ])
        
        return result
    
    def create_regions(self, num_regions=2):
        """
        Create a grid with distinct regions of different colors.
        
        Args:
            num_regions (int): Number of distinct regions to create
            
        Returns:
            list: Grid with distinct colored regions
        """
        grid = self.create_empty_grid(self.size)
        
        if num_regions == 2:
            # Simple vertical or horizontal split
            if random.choice([True, False]):
                # Vertical split
                split_col = random.randint(1, self.size - 2)
                color1, color2 = random.sample(range(1, 10), 2)
                
                for row in range(self.size):
                    for col in range(self.size):
                        if col <= split_col:
                            grid[row][col] = color1
                        else:
                            grid[row][col] = color2
            else:
                # Horizontal split
                split_row = random.randint(1, self.size - 2)
                color1, color2 = random.sample(range(1, 10), 2)
                
                for row in range(self.size):
                    for col in range(self.size):
                        if row <= split_row:
                            grid[row][col] = color1
                        else:
                            grid[row][col] = color2
        
        elif num_regions == 3:
            # Create an L-shaped region
            color1, color2, color3 = random.sample(range(1, 10), 3)
            
            # Fill with base color
            for row in range(self.size):
                for col in range(self.size):
                    grid[row][col] = color1
            
            # Create L-shape with color2
            for row in range(2, self.size):
                for col in range(2, self.size):
                    grid[row][col] = color2
            
            # Create small region with color3
            for row in range(0, 2):
                for col in range(3, self.size):
                    grid[row][col] = color3
        
        else:
            # Create a checkerboard-like pattern
            colors = random.sample(range(1, 10), min(num_regions, 4))
            
            for row in range(self.size):
                for col in range(self.size):
                    # Simple pattern based on position
                    region_id = (row // 2 + col // 2) % len(colors)
                    grid[row][col] = colors[region_id]
        
        return grid
    
    # not used right now 
    def create_fixed_pattern_examples(self, pattern_type):
        """
        Create examples with very specific, predictable patterns.
        
        Args:
            pattern_type (str): Type of pattern
            
        Returns:
            list: Three examples following the exact same rule
        """
        examples = []
        
        for i in range(3):
            if pattern_type == "always_fill_left":
                # Create vertical split grid
                input_grid = self.create_empty_grid(self.size)
                left_val = random.randint(1, 4)
                right_val = random.randint(5, 9)
                
                for row in range(self.size):
                    for col in range(self.size):
                        if col < self.size // 2:
                            input_grid[row][col] = left_val
                        else:
                            input_grid[row][col] = right_val
                
                # Always fill left side with value 7
                start_point = (0, 0)
                fill_value = 7
                output_grid = self.flood_fill(input_grid, start_point, fill_value)
                
            elif pattern_type == "increment_all":
                # Create any region pattern
                input_grid = self.create_regions(2)
                
                # Fill the entire grid by starting at (0,0) and incrementing
                start_point = (0, 0)
                current_val = input_grid[0][0]
                fill_value = min(current_val + 1, 9)
                output_grid = self.flood_fill(input_grid, start_point, fill_value)
                
            elif pattern_type == "fill_if_even":
                # Create grid with even and odd values
                input_grid = self.create_empty_grid(self.size)
                val1 = random.choice([2, 4, 6, 8])  # even
                val2 = random.choice([1, 3, 5, 7, 9])  # odd
                
                # Create pattern
                for row in range(self.size):
                    for col in range(self.size):
                        if (row + col) % 2 == 0:
                            input_grid[row][col] = val1
                        else:
                            input_grid[row][col] = val2
                
                # Find first even value cell
                start_point = None
                for row in range(self.size):
                    for col in range(self.size):
                        if input_grid[row][col] % 2 == 0:
                            start_point = (row, col)
                            break
                    if start_point:
                        break
                
                # Fill even regions with 9
                fill_value = 9
                output_grid = self.flood_fill(input_grid, start_point, fill_value)
            
            examples.append({
                'input': input_grid,
                'output': output_grid,
                'start_point': start_point,
                'fill_value': fill_value
            })
        
        return examples

    def create_shape_with_background(self):
        """
        Create a grid with a shape on a background color.
        
        Returns:
            tuple: (grid, shape_points, bg_color, shape_color)
        """
        grid = self.create_empty_grid(self.size)
        
        # Fill with background color
        bg_color = random.randint(1, 9)
        for row in range(self.size):
            for col in range(self.size):
                grid[row][col] = bg_color
        
        # Create a shape
        shape_color = random.choice([c for c in range(1, 10) if c != bg_color])
        shape_type = random.choice(["square", "plus", "L", "line"])
        
        if shape_type == "square":
            size = 2
            start_row = random.randint(1, self.size - size - 1)
            start_col = random.randint(1, self.size - size - 1)
            shape_points = []
            for dr in range(size):
                for dc in range(size):
                    grid[start_row + dr][start_col + dc] = shape_color
                    shape_points.append((start_row + dr, start_col + dc))
        
        elif shape_type == "plus":
            center_row = self.size // 2
            center_col = self.size // 2
            shape_points = [(center_row, center_col)]
            grid[center_row][center_col] = shape_color
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if 0 <= center_row + dr < self.size and 0 <= center_col + dc < self.size:
                    grid[center_row + dr][center_col + dc] = shape_color
                    shape_points.append((center_row + dr, center_col + dc))
        
        elif shape_type == "L":
            start_row = random.randint(0, self.size - 3)
            start_col = random.randint(0, self.size - 3)
            shape_points = []
            # Vertical part
            for dr in range(3):
                grid[start_row + dr][start_col] = shape_color
                shape_points.append((start_row + dr, start_col))
            # Horizontal part
            for dc in range(1, 3):
                grid[start_row + 2][start_col + dc] = shape_color
                shape_points.append((start_row + 2, start_col + dc))
        
        else:  # line
            if random.choice([True, False]):
                # Horizontal line
                row = random.randint(1, self.size - 2)
                shape_points = []
                for col in range(1, self.size - 1):
                    grid[row][col] = shape_color
                    shape_points.append((row, col))
            else:
                # Vertical line
                col = random.randint(1, self.size - 2)
                shape_points = []
                for row in range(1, self.size - 1):
                    grid[row][col] = shape_color
                    shape_points.append((row, col))
        
        return grid, shape_points, bg_color, shape_color
    
    def generate_flood_fill_example(self):
        """
        Generate a single flood fill example.
        
        Returns:
            dict: Example with input, output, and parameters
        """
        example_type = random.choice(["regions", "shape_background"])
        
        if example_type == "regions":
            # Create regions and fill one
            num_regions = random.randint(2, 3)
            input_grid = self.create_regions(num_regions)
            
            # Pick a random point to start flood fill
            start_row = random.randint(0, self.size - 1)
            start_col = random.randint(0, self.size - 1)
            
            # Pick a new color (different from current)
            current_color = input_grid[start_row][start_col]
            new_color = random.choice([c for c in range(1, 10) if c != current_color])
            
            # Apply flood fill
            output_grid = self.flood_fill(input_grid, (start_row, start_col), new_color)
            
        else:  # shape_background
            # Create shape on background, fill either shape or background
            input_grid, shape_points, bg_color, shape_color = self.create_shape_with_background()
            
            # Decide whether to fill shape or background
            if random.choice([True, False]):
                # Fill background
                # Find a background point
                bg_points = [(r, c) for r in range(self.size) for c in range(self.size) 
                             if input_grid[r][c] == bg_color]
                if bg_points:
                    start_row, start_col = random.choice(bg_points)
                else:
                    start_row, start_col = 0, 0
            else:
                # Fill shape
                if shape_points:
                    start_row, start_col = random.choice(shape_points)
                else:
                    start_row, start_col = self.size // 2, self.size // 2
            
            # Pick a new color
            current_color = input_grid[start_row][start_col]
            new_color = random.choice([c for c in range(1, 10) if c != current_color])
            
            # Apply flood fill
            output_grid = self.flood_fill(input_grid, (start_row, start_col), new_color)
        
        return {
            "input": input_grid,
            "output": output_grid,
            "start_point": (start_row, start_col),
            "fill_value": new_color
        }
    
    def create_fewshot_examples(self, num_examples=100):
        """
        Generate multiple few-shot examples with simple, consistent flood fill patterns.
        
        Args:
            num_examples (int): Number of examples to generate
            
        Returns:
            list: List of few-shot examples
        """
        examples = []
        
        for i in range(num_examples):
            # Simple pattern: always fill value X with value Y
            # Pick consistent values for all three examples
            target_value = random.randint(1, 9)  # Value to find and fill
            new_value = random.choice([v for v in range(1, 10) if v != target_value])  # Value to fill with
            
            examples_data = []
            
            for j in range(3):  # train1, train2, test
                # Create a grid with 2-3 regions
                input_grid = self.create_regions(random.randint(2, 3))
                
                # Find a cell with the target value
                start_point = None
                found = False
                
                # First, check if target_value exists
                has_target = False
                for row in range(self.size):
                    for col in range(self.size):
                        if input_grid[row][col] == target_value:
                            has_target = True
                            if not found:
                                start_point = (row, col)
                                found = True
                
                # If target value doesn't exist, replace one of the regions with it
                if not has_target:
                    # Get the color at position (0,0) and replace all instances with target_value
                    old_color = input_grid[0][0]
                    for row in range(self.size):
                        for col in range(self.size):
                            if input_grid[row][col] == old_color:
                                input_grid[row][col] = target_value
                    start_point = (0, 0)
                
                # Apply flood fill
                output_grid = self.flood_fill(input_grid, start_point, new_value)
                
                examples_data.append({
                    'input': input_grid,
                    'output': output_grid,
                    'start_point': start_point,
                    'fill_value': new_value
                })
            
            train1, train2, test = examples_data
            
            # In create_fewshot_examples method:
            solution = [
                f"output_grid = FloodFillGenerator.flood_fill(test_input, {test['start_point']}, {test['fill_value']})"
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