# count_and_transform.py
import random
import copy
from .base import ExampleGenerator

class CountAndTransformGenerator(ExampleGenerator):
    """
    A class to generate few-shot examples for the 'count and transform' task.
    """
    
    def __init__(self, size=5):
        """
        Initialize the CountAndTransformGenerator with a specified grid size.
        
        Args:
            size (int): The dimension of the square grid.
        """
        super().__init__(size)
    
    @staticmethod
    def count_shapes(grid):
        """
        Count the number of distinct non-zero shapes and create that many dots in a row.
        """
        if not grid or not grid[0]:
            return grid
        
        rows = len(grid)
        cols = len(grid[0])
        visited = [[False] * cols for _ in range(rows)]
        shape_count = 0
        
        def dfs(row, col, value):
            """Find connected component."""
            if (row < 0 or row >= rows or col < 0 or col >= cols or
                visited[row][col] or grid[row][col] != value or grid[row][col] == 0):
                return
            
            visited[row][col] = True
            # Check 4 neighbors
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                dfs(row + dr, col + dc, value)
        
        # Count distinct shapes
        for row in range(rows):
            for col in range(cols):
                if not visited[row][col] and grid[row][col] != 0:
                    dfs(row, col, grid[row][col])
                    shape_count += 1
        
        # Create output with dots
        output = [[0] * cols for _ in range(rows)]
        for i in range(min(shape_count, cols)):
            output[0][i] = 1  # Use color 1 for dots
        
        return output
    
    @staticmethod
    def mark_by_size(grid):
        """
        Replace each shape with a number (1-9) representing its size.
        """
        if not grid or not grid[0]:
            return grid
        
        rows = len(grid)
        cols = len(grid[0])
        visited = [[False] * cols for _ in range(rows)]
        output = [[0] * cols for _ in range(rows)]
        
        def dfs(row, col, value, cells):
            """Find connected component and collect cells."""
            if (row < 0 or row >= rows or col < 0 or col >= cols or
                visited[row][col] or grid[row][col] != value or grid[row][col] == 0):
                return
            
            visited[row][col] = True
            cells.append((row, col))
            
            # Check 4 neighbors
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                dfs(row + dr, col + dc, value, cells)
        
        # Find all shapes and mark by size
        for row in range(rows):
            for col in range(cols):
                if not visited[row][col] and grid[row][col] != 0:
                    cells = []
                    dfs(row, col, grid[row][col], cells)
                    
                    # Mark all cells with the size (capped at 9)
                    size = min(len(cells), 9)
                    for r, c in cells:
                        output[r][c] = size
        
        return output
    
    @staticmethod
    def mark_by_frequency(grid):
        """
        Mark the most frequent color with 1, all others with 0.
        """
        if not grid or not grid[0]:
            return grid
        
        rows = len(grid)
        cols = len(grid[0])
        
        # Count color frequencies
        counts = {}
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] != 0:
                    counts[grid[row][col]] = counts.get(grid[row][col], 0) + 1
        
        if not counts:
            return [[0] * cols for _ in range(rows)]
        
        # Find most frequent color
        most_frequent = max(counts, key=counts.get)
        
        # Create output marking most frequent color
        output = [[0] * cols for _ in range(rows)]
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == most_frequent:
                    output[row][col] = 1
        
        return output
    
    @staticmethod
    def count_colors(grid):
        """
        Create a horizontal line with length equal to number of distinct colors.
        """
        if not grid or not grid[0]:
            return grid
        
        rows = len(grid)
        cols = len(grid[0])
        
        # Count distinct colors
        colors = set()
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] != 0:
                    colors.add(grid[row][col])
        
        # Create output with horizontal line
        output = [[0] * cols for _ in range(rows)]
        color_count = len(colors)
        
        # Place dots in the middle row
        middle_row = rows // 2
        for i in range(min(color_count, cols)):
            output[middle_row][i] = 1
        
        return output
    
    def create_counting_pattern(self, pattern_type):
        """
        Create a grid with patterns suitable for counting operations.
        
        Args:
            pattern_type (str): Type of pattern to create
            
        Returns:
            list: Grid with pattern
        """
        grid = self.create_empty_grid(self.size)
        
        if pattern_type == "multiple_shapes":
            # Create 2-4 distinct shapes
            num_shapes = random.randint(2, 4)
            used_positions = set()
            
            for i in range(num_shapes):
                color = random.randint(1, 9)
                shape_size = random.randint(1, 3)
                
                # Try to place shape
                attempts = 0
                while attempts < 20:
                    start_row = random.randint(0, self.size - 1)
                    start_col = random.randint(0, self.size - 1)
                    
                    if (start_row, start_col) not in used_positions:
                        # Create small connected shape
                        positions = [(start_row, start_col)]
                        used_positions.add((start_row, start_col))
                        grid[start_row][start_col] = color
                        
                        # Try to expand
                        for _ in range(shape_size - 1):
                            if positions:
                                base = random.choice(positions)
                                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                    new_row, new_col = base[0] + dr, base[1] + dc
                                    if (0 <= new_row < self.size and 
                                        0 <= new_col < self.size and
                                        (new_row, new_col) not in used_positions):
                                        grid[new_row][new_col] = color
                                        used_positions.add((new_row, new_col))
                                        positions.append((new_row, new_col))
                                        break
                        break
                    attempts += 1
                    
        elif pattern_type == "frequency_pattern":
            # Create pattern with clear frequency differences
            colors = [1, 2, 3]
            frequencies = [5, 3, 1]  # Different frequencies
            
            positions = [(r, c) for r in range(self.size) for c in range(self.size)]
            random.shuffle(positions)
            
            idx = 0
            for color, freq in zip(colors, frequencies):
                for _ in range(min(freq, len(positions) - idx)):
                    if idx < len(positions):
                        row, col = positions[idx]
                        grid[row][col] = color
                        idx += 1
                        
        elif pattern_type == "sized_shapes":
            # Create shapes of different sizes
            sizes = [1, 2, 3, 4]
            random.shuffle(sizes)
            used_positions = set()
            
            for i, size in enumerate(sizes[:3]):  # Use up to 3 shapes
                color = i + 1
                
                # Find starting position
                attempts = 0
                while attempts < 20:
                    start_row = random.randint(0, self.size - 2)
                    start_col = random.randint(0, self.size - 2)
                    
                    if (start_row, start_col) not in used_positions:
                        # Create shape of specified size
                        positions = [(start_row, start_col)]
                        grid[start_row][start_col] = color
                        used_positions.add((start_row, start_col))
                        
                        # Expand to target size
                        while len(positions) < size:
                            if not positions:
                                break
                            base = random.choice(positions)
                            expanded = False
                            
                            # Try all directions
                            dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                            random.shuffle(dirs)
                            
                            for dr, dc in dirs:
                                new_row, new_col = base[0] + dr, base[1] + dc
                                if (0 <= new_row < self.size and 
                                    0 <= new_col < self.size and
                                    (new_row, new_col) not in used_positions):
                                    grid[new_row][new_col] = color
                                    used_positions.add((new_row, new_col))
                                    positions.append((new_row, new_col))
                                    expanded = True
                                    break
                            
                            if not expanded:
                                break
                        break
                    attempts += 1
                    
        elif pattern_type == "multiple_colors":
            # Create pattern with multiple distinct colors
            num_colors = random.randint(2, 4)
            colors = list(range(1, num_colors + 1))
            
            # Distribute colors randomly
            for row in range(self.size):
                for col in range(self.size):
                    if random.random() < 0.4:  # 40% chance of having a color
                        grid[row][col] = random.choice(colors)
        
        return grid
    
    def generate_count_example(self, count_type):
        """
        Generate a single counting example.
        
        Args:
            count_type (str): Type of counting transformation
            
        Returns:
            dict: Example with input and output
        """
        if count_type == "count_shapes":
            input_grid = self.create_counting_pattern("multiple_shapes")
            output_grid = self.count_shapes(input_grid)
            
        elif count_type == "mark_by_size":
            input_grid = self.create_counting_pattern("sized_shapes")
            output_grid = self.mark_by_size(input_grid)
            
        elif count_type == "mark_by_frequency":
            input_grid = self.create_counting_pattern("frequency_pattern")
            output_grid = self.mark_by_frequency(input_grid)
            
        elif count_type == "count_colors":
            input_grid = self.create_counting_pattern("multiple_colors")
            output_grid = self.count_colors(input_grid)
        
        return {
            "input": input_grid,
            "output": output_grid
        }
    
    def create_fewshot_examples(self, num_examples=100):
        """
        Generate multiple few-shot examples with consistent counting patterns.
        
        Args:
            num_examples (int): Number of examples to generate
            
        Returns:
            list: List of few-shot examples
        """
        examples = []
        
        # Pattern types to cycle through
        count_types = ["count_shapes", "mark_by_size", "mark_by_frequency", "count_colors"]
        
        for i in range(num_examples):
            count_type = count_types[i % len(count_types)]
            
            # Generate three examples with the same transformation
            train1 = self.generate_count_example(count_type)
            train2 = self.generate_count_example(count_type)
            test = self.generate_count_example(count_type)
            
            # Create solution based on count type
            if count_type == "count_shapes":
                solution = ["output_grid = CountAndTransformGenerator.count_shapes(test_input)"]
            elif count_type == "mark_by_size":
                solution = ["output_grid = CountAndTransformGenerator.mark_by_size(test_input)"]
            elif count_type == "mark_by_frequency":
                solution = ["output_grid = CountAndTransformGenerator.mark_by_frequency(test_input)"]
            elif count_type == "count_colors":
                solution = ["output_grid = CountAndTransformGenerator.count_colors(test_input)"]
            
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
    