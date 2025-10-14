import random
import copy
import math
from .base import ExampleGenerator

class GenerateBackgroundPatternGenerator(ExampleGenerator):
    """
    A class to generate few-shot examples for complex geometric background patterns.
    Handles creating patterns like diamonds, crosses, rings, and stars around a center point.
    """
    
    def __init__(self, size=5):
        """
        Initialize the GenerateBackgroundPatternGenerator with a specified grid size.
        
        Args:
            size (int): The dimension of the square grid.
        """
        super().__init__(size)
    
    @staticmethod
    def generate_background_pattern(grid, pattern_type="diamond", center=None, fill_color=2):
        """
        Generate a complex geometric pattern around a center point or hole.
        
        Args:
            grid (list): 2D grid with a background color and possibly a center hole
            pattern_type (str): Type of pattern to generate:
                - "diamond": Diamond/rhombus shape around center
                - "cross": Cross/plus shape around center
                - "ring": Concentric rings around center
                - "star": Star/asterisk shape around center
                - "checkerboard": Checkerboard pattern
                - "diagonal_stripes": Diagonal stripe pattern
            center (tuple): Center point (row, col). If None, auto-detect hole or use grid center
            fill_color (int): The color to use for filling the pattern (background)
            
        Returns:
            list: New grid with geometric pattern applied
        """
        if not grid or not grid[0]:
            return grid
        
        # Deep copy to avoid modifying original
        result = copy.deepcopy(grid)
        rows, cols = len(result), len(result[0])
        
        # Auto-detect center if not provided
        if center is None:
            # Look for a hole (cell with value 0 or different from most common)
            from collections import Counter
            all_values = [grid[i][j] for i in range(rows) for j in range(cols)]
            counts = Counter(all_values)
            
            # Find the hole (value different from most common)
            most_common_value = counts.most_common(1)[0][0]
            hole_found = False
            
            for i in range(rows):
                for j in range(cols):
                    if grid[i][j] != most_common_value:
                        center = (i, j)
                        hole_found = True
                        break
                if hole_found:
                    break
            
            # If no hole found, use grid center
            if not hole_found:
                center = (rows // 2, cols // 2)
        
        center_row, center_col = center
        
        if pattern_type == "diamond":
            # Diamond pattern: Keep cells on axes, remove diagonals at distance 1
            # Create diamond by removing cells based on Manhattan distance
            for i in range(rows):
                for j in range(cols):
                    if (i, j) == center:
                        continue  # Preserve center
                    
                    # Calculate distances
                    row_dist = abs(i - center_row)
                    col_dist = abs(j - center_col)
                    manhattan_dist = row_dist + col_dist
                    
                    # Diamond rule: Remove corners and diagonals
                    # Keep if on same row/col (cross arms)
                    on_axis = (i == center_row or j == center_col)
                    
                    # Check if on diagonal (equal row and col distance)
                    on_diagonal = (row_dist == col_dist and row_dist > 0)
                    
                    # Remove corners (max distance on both axes)
                    is_corner = (row_dist == max(abs(rows - 1 - center_row), center_row) and 
                                col_dist == max(abs(cols - 1 - center_col), center_col))
                    
                    # Pattern rules:
                    if on_diagonal or is_corner:
                        result[i][j] = 0  # Remove
                    # Keep axis cells with fill_color
        
        elif pattern_type == "cross":
            # Cross pattern: Keep only cells on same row or column as center
            for i in range(rows):
                for j in range(cols):
                    if (i, j) == center:
                        continue
                    
                    # Keep only if on same row or column
                    if i != center_row and j != center_col:
                        result[i][j] = 0
        
        elif pattern_type == "ring":
            # Ring pattern: Keep cells at specific distances from center
            target_distance = random.choice([1, 2])  # Ring radius
            
            for i in range(rows):
                for j in range(cols):
                    if (i, j) == center:
                        continue
                    
                    # Calculate Euclidean distance
                    distance = math.sqrt((i - center_row)**2 + (j - center_col)**2)
                    
                    # Keep if within distance range
                    if not (target_distance - 0.5 <= distance <= target_distance + 0.5):
                        result[i][j] = 0
        
        elif pattern_type == "star":
            # Star pattern: Keep cells on axes and diagonals (8-way)
            for i in range(rows):
                for j in range(cols):
                    if (i, j) == center:
                        continue
                    
                    row_dist = abs(i - center_row)
                    col_dist = abs(j - center_col)
                    
                    # Keep if on axis or diagonal
                    on_axis = (i == center_row or j == center_col)
                    on_diagonal = (row_dist == col_dist)
                    
                    if not (on_axis or on_diagonal):
                        result[i][j] = 0
        
        elif pattern_type == "checkerboard":
            # Checkerboard pattern relative to center
            for i in range(rows):
                for j in range(cols):
                    if (i, j) == center:
                        continue
                    
                    # Checkerboard: (row + col) % 2
                    if ((i - center_row) + (j - center_col)) % 2 == 0:
                        result[i][j] = 0
        
        elif pattern_type == "diagonal_stripes":
            # Diagonal stripe pattern
            for i in range(rows):
                for j in range(cols):
                    if (i, j) == center:
                        continue
                    
                    # Diagonal stripes: remove every other diagonal
                    if ((i - center_row) + (j - center_col)) % 2 == 0:
                        result[i][j] = 0
        
        elif pattern_type == "concentric_squares":
            # Concentric squares around center
            for i in range(rows):
                for j in range(cols):
                    if (i, j) == center:
                        continue
                    
                    # Chebyshev distance (max of row_dist and col_dist)
                    distance = max(abs(i - center_row), abs(j - center_col))
                    
                    # Remove every other square
                    if distance % 2 == 0:
                        result[i][j] = 0
        
        return result
    
    def create_uniform_grid_with_hole(self):
        """
        Create a grid filled with one color and a single hole.
        
        Returns:
            tuple: (grid, hole_position, fill_color)
        """
        grid = self.create_empty_grid(self.size)
        fill_color = random.randint(1, 9)
        
        # Fill entire grid
        for i in range(self.size):
            for j in range(self.size):
                grid[i][j] = fill_color
        
        # Create a hole at center or random position
        if random.random() < 0.7:
            # Center hole
            hole_row = self.size // 2
            hole_col = self.size // 2
        else:
            # Random hole (not too close to edges)
            hole_row = random.randint(1, self.size - 2)
            hole_col = random.randint(1, self.size - 2)
        
        grid[hole_row][hole_col] = 0
        
        return grid, (hole_row, hole_col), fill_color
    
    def create_grid_with_background(self):
        """
        Create a grid with a uniform background color.
        
        Returns:
            tuple: (grid, center_position, fill_color)
        """
        grid = self.create_empty_grid(self.size)
        fill_color = random.randint(1, 9)
        
        # Fill entire grid
        for i in range(self.size):
            for j in range(self.size):
                grid[i][j] = fill_color
        
        # Use center position
        center = (self.size // 2, self.size // 2)
        
        # Optionally create a center hole
        if random.random() < 0.5:
            grid[center[0]][center[1]] = 0
        
        return grid, center, fill_color
    
    def generate_background_pattern_example(self, pattern_type=None):
        """
        Generate a single background pattern example.
        
        Args:
            pattern_type (str): Type of pattern. If None, chosen randomly.
            
        Returns:
            dict: Example with input, output, and parameters
        """
        if pattern_type is None:
            pattern_type = random.choice([
                "diamond",
                "cross",
                "ring",
                "star",
                "checkerboard",
                "diagonal_stripes",
                "concentric_squares"
            ])
        
        # Create input grid with hole or uniform background
        if random.random() < 0.6:
            input_grid, center, fill_color = self.create_uniform_grid_with_hole()
        else:
            input_grid, center, fill_color = self.create_grid_with_background()
        
        # Apply pattern
        output_grid = self.generate_background_pattern(input_grid, pattern_type, center, fill_color)
        
        return {
            "input": input_grid,
            "output": output_grid,
            "pattern_type": pattern_type,
            "center": center,
            "fill_color": fill_color
        }
    
    def create_fewshot_examples(self, num_examples=100):
        """
        Generate multiple few-shot examples with consistent pattern type.
        
        Args:
            num_examples (int): Number of examples to generate
            
        Returns:
            list: List of few-shot examples
        """
        examples = []
        
        for i in range(num_examples):
            # Pick a consistent pattern type for all three examples
            pattern_type = random.choice([
                "diamond",
                "cross",
                "ring",
                "star",
                "checkerboard",
                "diagonal_stripes",
                "concentric_squares"
            ])
            
            # Pick a consistent fill color
            fill_color = random.randint(1, 9)
            
            # Generate three examples with same pattern type
            examples_data = []
            for j in range(3):
                # Create input grid with hole or uniform background
                if random.random() < 0.6:
                    input_grid, center, _ = self.create_uniform_grid_with_hole()
                else:
                    input_grid, center, _ = self.create_grid_with_background()
                
                # Override with consistent fill color
                for row_idx in range(self.size):
                    for col_idx in range(self.size):
                        if input_grid[row_idx][col_idx] != 0:
                            input_grid[row_idx][col_idx] = fill_color
                
                # Apply pattern
                output_grid = self.generate_background_pattern(input_grid, pattern_type, center, fill_color)
                
                examples_data.append({
                    "input": input_grid,
                    "output": output_grid,
                    "pattern_type": pattern_type,
                    "center": center,
                    "fill_color": fill_color
                })
            
            train1, train2, test = examples_data
            
            # Create solution - handle tuple center properly for code generation
            center_str = f"({test['center'][0]}, {test['center'][1]})"
            solution = [
                f"output_grid = GenerateBackgroundPatternGenerator.generate_background_pattern(test_input, '{test['pattern_type']}', {center_str}, {test['fill_color']})"
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
