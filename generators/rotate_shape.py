import random
import copy
from .base import ExampleGenerator

class RotateShapeGenerator(ExampleGenerator):
    """
    A class to generate few-shot examples for the 'rotate shape' task.
    """
    
    def __init__(self, size=5):
        """
        Initialize the RotateShapeGenerator with a specified grid size.
        
        Args:
            size (int): The dimension of the square grid.
        """
        super().__init__(size)
    
    @staticmethod
    def rotate(points, center, times, direction):
        """
        Rotate a set of points around a center point.
        
        Args:
            points (list): List of (row, col) tuples representing the shape
            center (tuple): (row, col) center of rotation
            times (int): Number of 90-degree rotations (1, 2, or 3)
            direction (str): 'clockwise' or 'counterclockwise'
            
        Returns:
            list: List of rotated (row, col) tuples
        """
        rotated_points = []
        center_row, center_col = center
        
        # Convert times and direction to angle
        angle = (times * 90) % 360
        if direction == 'counterclockwise':
            angle = 360 - angle
        
        for row, col in points:
            # Translate point to origin
            translated_row = row - center_row
            translated_col = col - center_col
            
            # Rotate around origin
            if angle == 90:
                new_row = translated_col
                new_col = -translated_row
            elif angle == 180:
                new_row = -translated_row
                new_col = -translated_col
            elif angle == 270:
                new_row = -translated_col
                new_col = translated_row
            else:  # 0 or 360
                new_row = translated_row
                new_col = translated_col
            
            # Translate back
            final_row = new_row + center_row
            final_col = new_col + center_col
            
            rotated_points.append((final_row, final_col))
        
        return rotated_points
    
    @staticmethod
    def rotate_clockwise(grid):
        """
        Rotate all shapes in the grid 90 degrees clockwise around their center.
        
        Args:
            grid (list): Input grid
            
        Returns:
            list: New grid with rotated shape
        """
        # Detect shape
        shape_coords = []
        value = None
        
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] != 0:
                    shape_coords.append((row, col))
                    if value is None:
                        value = grid[row][col]
        
        if not shape_coords:
            return grid
        
        # Calculate center
        center_row = sum(coord[0] for coord in shape_coords) / len(shape_coords)
        center_col = sum(coord[1] for coord in shape_coords) / len(shape_coords)
        center_row = round(center_row)
        center_col = round(center_col)
        
        # Create new grid
        new_grid = [[0 for _ in range(len(grid[0]))] for _ in range(len(grid))]
        
        # Rotate points
        rotated_points = RotateShapeGenerator.rotate(shape_coords, (center_row, center_col), 1, 'clockwise')
        
        # Place rotated points
        for row, col in rotated_points:
            if 0 <= row < len(grid) and 0 <= col < len(grid[0]):
                new_grid[row][col] = value
        
        return new_grid

    @staticmethod
    def rotate_counterclockwise(grid):
        """
        Rotate all shapes in the grid 90 degrees counterclockwise around their center.
        
        Args:
            grid (list): Input grid
            
        Returns:
            list: New grid with rotated shape
        """
        # Detect shape
        shape_coords = []
        value = None
        
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] != 0:
                    shape_coords.append((row, col))
                    if value is None:
                        value = grid[row][col]
        
        if not shape_coords:
            return grid
        
        # Calculate center
        center_row = sum(coord[0] for coord in shape_coords) / len(shape_coords)
        center_col = sum(coord[1] for coord in shape_coords) / len(shape_coords)
        center_row = round(center_row)
        center_col = round(center_col)
        
        # Create new grid
        new_grid = [[0 for _ in range(len(grid[0]))] for _ in range(len(grid))]
        
        # Rotate points
        rotated_points = RotateShapeGenerator.rotate(shape_coords, (center_row, center_col), 1, 'counterclockwise')
        
        # Place rotated points
        for row, col in rotated_points:
            if 0 <= row < len(grid) and 0 <= col < len(grid[0]):
                new_grid[row][col] = value
        
        return new_grid
    @staticmethod
    def detect_shape(grid):
        """
        Detect all non-zero cells in the grid as a shape.
        
        Args:
            grid (list): 2D grid
            
        Returns:
            tuple: (shape_coords, value, center) where shape_coords is list of (row, col),
                value is the shape's value, and center is (row, col) of shape's center
        """
        shape_coords = []
        value = None
        
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] != 0:
                    shape_coords.append((row, col))
                    if value is None:
                        value = grid[row][col]
        
        if not shape_coords:
            return [], None, None
        
        # Calculate center of shape (using average of coordinates)
        center_row = sum(coord[0] for coord in shape_coords) / len(shape_coords)
        center_col = sum(coord[1] for coord in shape_coords) / len(shape_coords)
        
        # Round to nearest integer for grid positioning
        center_row = round(center_row)
        center_col = round(center_col)
        
        return shape_coords, value, (center_row, center_col)
    
    @staticmethod
    def rotate_shape_on_grid(grid, center_row, center_col, times, direction):
        """
        Rotate a shape on a grid using the rotate function.
        
        Args:
            grid (list): Input grid
            center_row (int): Row coordinate of rotation center
            center_col (int): Column coordinate of rotation center
            times (int): Number of 90-degree rotations
            direction (str): 'clockwise' or 'counterclockwise'
            
        Returns:
            list: New grid with rotated shape
        """
        shape_coords, value, _ = RotateShapeGenerator.detect_shape(grid)
        if not shape_coords:
            return grid
        
        # Create new grid with correct dimensions
        new_grid = [[0 for _ in range(len(grid[0]))] for _ in range(len(grid))]
        center = (center_row, center_col)
        
        # Rotate the shape points
        rotated_points = RotateShapeGenerator.rotate(shape_coords, center, times, direction)
        
        # Place rotated points on new grid
        for row, col in rotated_points:
            # Check if new position is within grid bounds
            if 0 <= row < len(new_grid) and 0 <= col < len(new_grid[0]):
                new_grid[row][col] = value
        
        return new_grid
    
    def generate_random_shape(self, max_size=3):
        """
        Generate a random connected shape.
        
        Returns:
            tuple: (shape_coords, center_row, center_col)
        """
        # Adjust max_size based on grid size
        max_size = min(max_size, self.size - 2)  # Ensure shape fits with margins
        
        # Start with a single cell
        shape = [(0, 0)]
        
        # Randomly add adjacent cells
        num_cells = min(random.randint(2, max_size), 4)  # Limit to 4 cells max
        
        for _ in range(num_cells - 1):
            if not shape:
                break
            
            # Pick a random cell from the shape
            base_cell = random.choice(shape)
            
            # Pick a random direction
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(directions)
            
            # Try each direction until we find a valid cell
            for dr, dc in directions:
                new_cell = (base_cell[0] + dr, base_cell[1] + dc)
                
                # Add if not already in shape
                if new_cell not in shape:
                    shape.append(new_cell)
                    break
        
        # Calculate bounds
        if shape:
            min_r = min(r for r, c in shape)
            min_c = min(c for r, c in shape)
            max_r = max(r for r, c in shape)
            max_c = max(c for r, c in shape)
        else:
            shape = [(0, 0)]
            min_r = min_c = max_r = max_c = 0
        
        # Normalize to start at (0, 0)
        shape = [(r - min_r, c - min_c) for r, c in shape]
        
        # Calculate center
        shape_height = max_r - min_r
        shape_width = max_c - min_c
        center_r = shape_height // 2
        center_c = shape_width // 2
        
        return shape, center_r, center_c
    
    
    def generate_rotate_example(self):
        """
        Generate a single rotation example.
        
        Returns:
            dict: Example with input, output, and parameters
        """
        # Generate a random shape
        shape_template, shape_center_r, shape_center_c = self.generate_random_shape()
        
        # Calculate shape bounds
        max_r = max(r for r, c in shape_template)
        max_c = max(c for r, c in shape_template)
        
        # Calculate available space
        # We need space for the shape plus room for rotation
        # The worst case is a 90-degree rotation which could extend the shape
        shape_size = max(max_r + 1, max_c + 1)
        
        # Ensure we have enough space on the grid
        if shape_size > self.size - 1:
            # If shape is too big, scale it down
            shape_template = [(0, 0)]  # Use single cell as fallback
            shape_center_r = 0
            shape_center_c = 0
            max_r = 0
            max_c = 0
        
        # Calculate safe placement bounds
        # Leave at least 1 cell margin on all sides for rotation
        min_start = 1
        max_start_row = max(min_start, self.size - max_r - 2)
        max_start_col = max(min_start, self.size - max_c - 2)
        
        # Place shape randomly on grid
        start_row = random.randint(min_start, max_start_row)
        start_col = random.randint(min_start, max_start_col)
        
        # Create input grid
        input_grid = self.create_empty_grid(self.size)
        value = random.randint(1, 9)
        
        # Place shape on grid
        shape_coords = []
        for r, c in shape_template:
            grid_row = start_row + r
            grid_col = start_col + c
            input_grid[grid_row][grid_col] = value
            shape_coords.append((grid_row, grid_col))
        
        # Determine rotation center (shape center in grid coordinates)
        center_row = start_row + shape_center_r
        center_col = start_col + shape_center_c
        
        # Choose random rotation parameters
        times = random.randint(1, 3)
        direction = random.choice(['clockwise', 'counterclockwise'])
        
        # Create output grid
        output_grid = self.rotate_shape_on_grid(input_grid, center_row, center_col, times, direction)
        
        return {
            "input": input_grid,
            "output": output_grid,
            "center": (center_row, center_col),
            "times": times,
            "direction": direction,
            "value": value,
            "shape_coords": shape_coords
        }
    
    def create_fewshot_examples(self, num_examples=100):
        """
        Generate multiple few-shot examples with two training pairs and one test pair.
        All three use the same rotation parameters (times and direction).
        
        Args:
            num_examples (int): Number of examples to generate
            
        Returns:
            list: List of few-shot examples
        """
        examples = []
        
        for i in range(num_examples):
            # For consistency, use a fixed rotation pattern
            # Always rotate around the center of the grid
            grid_center_row = self.size // 2
            grid_center_col = self.size // 2
            
            # Generate first training example
            train1 = self.generate_rotate_example()
            
            # Extract rotation parameters from train1
            times = train1['times']
            direction = train1['direction']
            
            # Override the output to use grid center rotation
            train1['output'] = self.rotate_shape_on_grid(
                train1['input'], 
                grid_center_row,  # Use grid center instead of shape center
                grid_center_col, 
                times, 
                direction
            )
            
            # Generate second training example with same rotation
            train2 = self.generate_rotate_example()
            train2['times'] = times
            train2['direction'] = direction
            train2['output'] = self.rotate_shape_on_grid(
                train2['input'], 
                grid_center_row,  # Use grid center
                grid_center_col, 
                times, 
                direction
            )
            
            # Generate test example with same rotation
            test = self.generate_rotate_example()
            test['times'] = times
            test['direction'] = direction
            test['output'] = self.rotate_shape_on_grid(
                test['input'], 
                grid_center_row,  # Use grid center
                grid_center_col, 
                times, 
                direction
            )
            
            # Create solution using fixed grid center
            solution = [
                f"output_grid = RotateShapeGenerator.rotate_shape_on_grid(test_input, {grid_center_row}, {grid_center_col}, {times}, '{direction}')"
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