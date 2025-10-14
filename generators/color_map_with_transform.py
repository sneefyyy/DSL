import random
import copy
from .base import ExampleGenerator

class ColorMapWithTransformGenerator(ExampleGenerator):
    """
    A class to generate few-shot examples for simultaneous color mapping and spatial transformation.
    This handles cases where objects must change color AND position at the same time.
    """
    
    def __init__(self, size=5):
        """
        Initialize the ColorMapWithTransformGenerator with a specified grid size.
        
        Args:
            size (int): The dimension of the square grid.
        """
        super().__init__(size)
    
    @staticmethod
    def color_map_with_transform(grid, color_mapping, spatial_transform):
        """
        Apply color mapping and spatial transformation simultaneously to preserve object identity.
        
        Args:
            grid (list): 2D grid with objects (non-zero values)
            color_mapping (dict): Map of old_color -> new_color (e.g., {8: 2, 5: 3})
            spatial_transform (dict): Spatial transformation specification:
                - {"shift": {"dx": 0, "dy": 1}}: Shift by dx columns, dy rows
                - {"shift": {"dx": 1, "dy": 0}, "region": {"top_left": (0,0), "bottom_right": (1,1)}}: Shift only region
                - {"rotate": {"angle": 90}}: Rotate 90/180/270 degrees clockwise
                - {"flip": {"axis": "horizontal"}}: Mirror horizontally or vertically
            
        Returns:
            list: New grid with simultaneous transforms applied
        """
        if not grid or not grid[0]:
            return grid
        
        rows, cols = len(grid), len(grid[0])
        result = [[0 for _ in range(cols)] for _ in range(rows)]
        
        # Extract region if specified, otherwise use entire grid
        region = spatial_transform.get("region", None)
        if region:
            top_left = region.get("top_left", (0, 0))
            bottom_right = region.get("bottom_right", (rows-1, cols-1))
            region_coords = [(i, j) for i in range(top_left[0], bottom_right[0] + 1)
                           for j in range(top_left[1], bottom_right[1] + 1)]
        else:
            region_coords = [(i, j) for i in range(rows) for j in range(cols)]
        
        # Determine transform type
        if "shift" in spatial_transform:
            # Shift transform
            dx = spatial_transform["shift"].get("dx", 0)
            dy = spatial_transform["shift"].get("dy", 0)
            
            for i, j in region_coords:
                if grid[i][j] != 0:
                    # Apply color mapping
                    new_color = color_mapping.get(grid[i][j], grid[i][j])
                    # Apply spatial shift
                    new_i = i + dy
                    new_j = j + dx
                    # Place if in bounds
                    if 0 <= new_i < rows and 0 <= new_j < cols:
                        result[new_i][new_j] = new_color
        
        elif "rotate" in spatial_transform:
            # Rotate transform (around center of region or grid)
            angle = spatial_transform["rotate"].get("angle", 90)
            
            # Calculate center of rotation
            if region:
                center_i = (region["top_left"][0] + region["bottom_right"][0]) / 2
                center_j = (region["top_left"][1] + region["bottom_right"][1]) / 2
            else:
                center_i = (rows - 1) / 2
                center_j = (cols - 1) / 2
            
            for i, j in region_coords:
                if grid[i][j] != 0:
                    # Apply color mapping
                    new_color = color_mapping.get(grid[i][j], grid[i][j])
                    
                    # Rotate coordinates around center
                    rel_i = i - center_i
                    rel_j = j - center_j
                    
                    if angle == 90:
                        new_rel_i = rel_j
                        new_rel_j = -rel_i
                    elif angle == 180:
                        new_rel_i = -rel_i
                        new_rel_j = -rel_j
                    elif angle == 270:
                        new_rel_i = -rel_j
                        new_rel_j = rel_i
                    else:
                        new_rel_i = rel_i
                        new_rel_j = rel_j
                    
                    new_i = int(round(center_i + new_rel_i))
                    new_j = int(round(center_j + new_rel_j))
                    
                    # Place if in bounds
                    if 0 <= new_i < rows and 0 <= new_j < cols:
                        result[new_i][new_j] = new_color
        
        elif "flip" in spatial_transform:
            # Flip transform
            axis = spatial_transform["flip"].get("axis", "horizontal")
            
            # Calculate flip axis
            if region:
                if axis == "horizontal":
                    flip_axis = (region["top_left"][1] + region["bottom_right"][1]) / 2
                else:  # vertical
                    flip_axis = (region["top_left"][0] + region["bottom_right"][0]) / 2
            else:
                if axis == "horizontal":
                    flip_axis = (cols - 1) / 2
                else:  # vertical
                    flip_axis = (rows - 1) / 2
            
            for i, j in region_coords:
                if grid[i][j] != 0:
                    # Apply color mapping
                    new_color = color_mapping.get(grid[i][j], grid[i][j])
                    
                    # Apply flip
                    if axis == "horizontal":
                        new_i = i
                        new_j = int(round(2 * flip_axis - j))
                    else:  # vertical
                        new_i = int(round(2 * flip_axis - i))
                        new_j = j
                    
                    # Place if in bounds
                    if 0 <= new_i < rows and 0 <= new_j < cols:
                        result[new_i][new_j] = new_color
        
        return result
    
    def create_simple_shape(self, shape_type="block"):
        """
        Create a simple shape at a random position.
        
        Args:
            shape_type (str): Type of shape - "block", "line_h", "line_v", "L"
            
        Returns:
            tuple: (grid, shape_positions, shape_color)
        """
        grid = self.create_empty_grid(self.size)
        color = random.randint(1, 9)
        positions = []
        
        if shape_type == "block":
            # 2x2 block
            max_row = self.size - 2
            max_col = self.size - 2
            if max_row > 0 and max_col > 0:
                start_row = random.randint(0, max_row)
                start_col = random.randint(0, max_col)
                for i in range(2):
                    for j in range(2):
                        grid[start_row + i][start_col + j] = color
                        positions.append((start_row + i, start_col + j))
        
        elif shape_type == "line_h":
            # Horizontal line of length 3
            row = random.randint(0, self.size - 1)
            start_col = random.randint(0, self.size - 3)
            for j in range(3):
                grid[row][start_col + j] = color
                positions.append((row, start_col + j))
        
        elif shape_type == "line_v":
            # Vertical line of length 3
            col = random.randint(0, self.size - 1)
            start_row = random.randint(0, self.size - 3)
            for i in range(3):
                grid[start_row + i][col] = color
                positions.append((start_row + i, col))
        
        elif shape_type == "L":
            # L-shape
            max_row = self.size - 3
            max_col = self.size - 3
            if max_row >= 0 and max_col >= 0:
                start_row = random.randint(0, max(0, max_row))
                start_col = random.randint(0, max(0, max_col))
                # Vertical part
                for i in range(3):
                    grid[start_row + i][start_col] = color
                    positions.append((start_row + i, start_col))
                # Horizontal part
                for j in range(1, 3):
                    grid[start_row + 2][start_col + j] = color
                    positions.append((start_row + 2, start_col + j))
        
        return grid, positions, color
    
    def generate_color_map_transform_example(self, transform_type=None):
        """
        Generate a single color_map_with_transform example.
        
        Args:
            transform_type (str): Type of transform. If None, chosen randomly.
            
        Returns:
            dict: Example with input, output, and parameters
        """
        if transform_type is None:
            transform_type = random.choice(["shift", "rotate", "flip"])
        
        # Create a simple shape
        shape_type = random.choice(["block", "line_h", "line_v", "L"])
        input_grid, positions, old_color = self.create_simple_shape(shape_type)
        
        # Create color mapping
        new_color = random.choice([c for c in range(1, 10) if c != old_color])
        color_mapping = {old_color: new_color}
        
        # Create spatial transform
        if transform_type == "shift":
            # Random shift that keeps shape in bounds
            dx = random.randint(-1, 1)
            dy = random.randint(-1, 1)
            if dx == 0 and dy == 0:
                dy = 1  # Ensure some movement
            
            spatial_transform = {"shift": {"dx": dx, "dy": dy}}
            
            # Optionally add region constraint
            if random.random() < 0.5 and positions:
                min_row = min(p[0] for p in positions)
                max_row = max(p[0] for p in positions)
                min_col = min(p[1] for p in positions)
                max_col = max(p[1] for p in positions)
                spatial_transform["region"] = {
                    "top_left": (min_row, min_col),
                    "bottom_right": (max_row, max_col)
                }
        
        elif transform_type == "rotate":
            angle = random.choice([90, 180, 270])
            spatial_transform = {"rotate": {"angle": angle}}
            
            # Add region constraint for the shape
            if positions:
                min_row = min(p[0] for p in positions)
                max_row = max(p[0] for p in positions)
                min_col = min(p[1] for p in positions)
                max_col = max(p[1] for p in positions)
                spatial_transform["region"] = {
                    "top_left": (min_row, min_col),
                    "bottom_right": (max_row, max_col)
                }
        
        elif transform_type == "flip":
            axis = random.choice(["horizontal", "vertical"])
            spatial_transform = {"flip": {"axis": axis}}
            
            # Add region constraint for the shape
            if positions:
                min_row = min(p[0] for p in positions)
                max_row = max(p[0] for p in positions)
                min_col = min(p[1] for p in positions)
                max_col = max(p[1] for p in positions)
                spatial_transform["region"] = {
                    "top_left": (min_row, min_col),
                    "bottom_right": (max_row, max_col)
                }
        
        # Apply transform
        output_grid = self.color_map_with_transform(input_grid, color_mapping, spatial_transform)
        
        return {
            "input": input_grid,
            "output": output_grid,
            "color_mapping": color_mapping,
            "spatial_transform": spatial_transform
        }
    
    def create_fewshot_examples(self, num_examples=100):
        """
        Generate multiple few-shot examples with consistent transform patterns.
        
        Args:
            num_examples (int): Number of examples to generate
            
        Returns:
            list: List of few-shot examples
        """
        examples = []
        
        for i in range(num_examples):
            # Pick consistent transform type
            transform_type = random.choice(["shift", "rotate", "flip"])
            
            # Pick consistent color mapping pattern (same colors across examples)
            old_color = random.randint(1, 9)
            new_color = random.choice([c for c in range(1, 10) if c != old_color])
            color_mapping = {old_color: new_color}
            
            # Generate consistent spatial transform parameters
            if transform_type == "shift":
                dx = random.randint(-1, 1)
                dy = random.randint(-1, 1)
                if dx == 0 and dy == 0:
                    dy = 1
                spatial_params = {"shift": {"dx": dx, "dy": dy}}
            elif transform_type == "rotate":
                angle = random.choice([90, 180, 270])
                spatial_params = {"rotate": {"angle": angle}}
            else:  # flip
                axis = random.choice(["horizontal", "vertical"])
                spatial_params = {"flip": {"axis": axis}}
            
            # Generate three examples with same transform but different shapes
            examples_data = []
            for j in range(3):
                shape_type = random.choice(["block", "line_h", "line_v", "L"])
                input_grid, positions, _ = self.create_simple_shape(shape_type)
                
                # Override with our chosen color
                for pos in positions:
                    input_grid[pos[0]][pos[1]] = old_color
                
                # Apply same transform
                spatial_transform = copy.deepcopy(spatial_params)
                
                # Add region if needed
                if positions and transform_type in ["rotate", "flip"]:
                    min_row = min(p[0] for p in positions)
                    max_row = max(p[0] for p in positions)
                    min_col = min(p[1] for p in positions)
                    max_col = max(p[1] for p in positions)
                    spatial_transform["region"] = {
                        "top_left": (min_row, min_col),
                        "bottom_right": (max_row, max_col)
                    }
                
                output_grid = self.color_map_with_transform(input_grid, color_mapping, spatial_transform)
                
                examples_data.append({
                    "input": input_grid,
                    "output": output_grid,
                    "color_mapping": color_mapping,
                    "spatial_transform": spatial_transform
                })
            
            train1, train2, test = examples_data
            
            # Create solution
            solution = [
                f"output_grid = ColorMapWithTransformGenerator.color_map_with_transform(test_input, {test['color_mapping']}, {test['spatial_transform']})"
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
