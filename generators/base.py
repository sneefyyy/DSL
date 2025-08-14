import json
from abc import ABC, abstractmethod

class ExampleGenerator(ABC):
    """
    A general base class for generating various types of examples.
    It provides common attributes and utility methods for grid manipulation
    and line validation that specific example generators can inherit and use.
    """
    def __init__(self, size=5):
        """
        Initializes the ExampleGenerator with a specified grid size.

        Args:
            size (int): The default dimension of the square grid for generated examples.
        """
        self.size = size

    def create_empty_grid(self, size=None):
        """
        Creates an empty square grid filled with zeros.

        Args:
            size (int, optional): The dimension of the square grid.
                                  Defaults to self.size if not provided.

        Returns:
            list: A 2D list representing the empty grid.
        """
        actual_size = size if size is not None else self.size
        return [[0] * actual_size for _ in range(actual_size)]

    @staticmethod
    def make_shape(points, value, grid):
        """
        Fills specified points in an existing grid with a given value.

        Args:
            points (list): A list of (row, column, value) tuples.
                           The 'value' in the tuple is ignored, as 'value' argument is used.
            value (int): The value to set at the specified points.
            grid (list): The 2D list representing the grid to modify.
        """
        for r, c, _ in points:
            grid[r][c] = value

    @staticmethod
    def is_valid_line(start, end):
        """
        Checks if a line segment between two points is horizontal, vertical, or diagonal.

        Args:
            start (tuple): The starting (x, y) coordinate.
            end (tuple): The ending (x, y) coordinate.

        Returns:
            bool: True if the line is horizontal, vertical, or diagonal; False otherwise.
        """
        x1, y1 = start
        x2, y2 = end
        return (
            x1 == x2 or  # Horizontal line
            y1 == y2 or  # Vertical line
            abs(x2 - x1) == abs(y2 - y1)  # Diagonal line 
        )
    
    @abstractmethod
    def create_fewshot_examples(self, num_examples):
        """
        Abstract method to generate a list of few-shot examples.
        Concrete subclasses must implement this.

        Args:
            num_examples (int): The number of examples to generate.

        Returns:
            list: A list of dictionaries, where each dictionary is a few-shot example.
        """
        pass

    def save_fewshot_examples(self, filename, num_examples=100):
        """
        Generates a specified number of few-shot examples (by calling
        the concrete implementation of create_fewshot_examples) and saves
        them to a JSON file.

        Args:
            filename (str): The name of the file (e.g., "my_examples.json")
                            to save the generated data.
            num_examples (int): The total number of examples to generate and save.
        """
        data = self.create_fewshot_examples(num_examples)
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Finished writing {num_examples} examples to {filename}")
