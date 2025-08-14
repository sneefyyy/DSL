# Connect.py
import random
import copy
from .base import ExampleGenerator

class ConnectGenerator(ExampleGenerator):
    """
    A class to generate few-shot examples for the 'connect' task.
    It encapsulates the logic for creating grids, connecting points,
    and structuring the training and testing examples. Inherits from ExampleGenerator.
    """

    def __init__(self, size=5):
        """
        Initializes the ConnectSegmentGenerator with a specified grid size,
        calling the base class constructor.

        Args:
            size (int): The default dimension of the square grid for generated examples.
        """
        super().__init__(size) 

    @staticmethod
    def connect(grid, start, end, value):
        """
        Connect two points on a grid with a straight line and return the modified grid.
        Supports horizontal, vertical, and diagonal lines.

        Args:
            grid (list): The input grid to modify
            start (tuple): The starting (row, column) coordinate.
            end (tuple): The ending (row, column) coordinate.
            value (int): The value to assign to each point along the line.

        Returns:
            list: The modified grid with the connection applied.
        """
        # Deep copy to avoid modifying original
        result = copy.deepcopy(grid)
        
        x1, y1 = start
        x2, y2 = end
        dx = (x2 - x1) and ((x2 - x1) // abs(x2 - x1))
        dy = (y2 - y1) and ((y2 - y1) // abs(y2 - y1))

        x, y = x1, y1
        while True:
            if 0 <= x < len(result) and 0 <= y < len(result[0]):
                result[x][y] = value
            if (x, y) == (x2, y2): 
                break
            x += dx
            y += dy
            
        return result

    def apply_connection_to_grid(self, points, size=None):
        """
        Creates a new empty grid and then applies a list of (row, column, value)
        points to it, filling the specified cells with their respective values.

        Args:
            points (list): A list of (row, column, value) tuples.
            size (int, optional): The size of the grid to create and apply the connection to.
                                  Defaults to self.size if not provided.

        Returns:
            list: A 2D list representing the grid with the connection applied.
        """
        grid = self.create_empty_grid(size)
        for i, j, v in points:
            grid[i][j] = v
        return grid

    def generate_connect_example(self, with_input_markers=True):
        """
        Generates a single 'connect' example, including the input grid (with optional
        start/end markers), the expected output grid (with the full connection),
        and the parameters needed to generate the solution (start, end, value).

        Args:
            with_input_markers (bool): If True, the input grid will have the 'value'
                                       placed only at the start and end points.
                                       If False, the input grid will be entirely empty.

        Returns:
            dict: A dictionary containing the generated example data.
        """
        while True:
            start = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            end = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if start != end and self.is_valid_line(start, end):
                break

        value = random.randint(1, 9)
        
        # Create input grid with markers
        input_grid = self.create_empty_grid(self.size)
        if with_input_markers:
            input_grid[start[0]][start[1]] = value
            input_grid[end[0]][end[1]] = value

        # Create output grid using the new connect method
        output_grid = self.connect(input_grid, start, end, value)

        return {
            "input": input_grid,
            "output": output_grid,
            "start": start,
            "end": end,
            "value": value
        }

    def create_fewshot_examples(self, num_examples=100):
        """
        Generates a list of multiple few-shot examples, each containing
        training pairs and a testing pair with a corresponding solution.

        Args:
            num_examples (int): The number of examples to generate.

        Returns:
            list: A list of dictionaries, where each dictionary is a few-shot example.
        """
        examples = []
        for i in range(num_examples):
            # Generate two training examples and one test example
            train1 = self.generate_connect_example()
            train2 = self.generate_connect_example()
            test = self.generate_connect_example()


            solution = [
                f"output_grid = ConnectGenerator.connect(test_input, {test['start']}, {test['end']}, {test['value']})"
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