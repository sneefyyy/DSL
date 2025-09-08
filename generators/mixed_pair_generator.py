import random
import copy
from .base import ExampleGenerator
from .mirror_shape import MirrorShapeGenerator
from .rotate_shape import RotateShapeGenerator
from .symmetry_complete_generator import SymmetryCompleteGenerator
from .count_and_transform_generator import CountAndTransformGenerator


class MixedPairGenerator(ExampleGenerator):
    """Generator that composes ANY TWO transformations sequentially.

    For each example we:
      1. Randomly pick two transformation primitives (with replacement) from a pool
         of static methods that each accept a grid and return a transformed grid.
      2. Sample three independent random input grids (train1, train2, test).
      3. Apply transform A then transform B to each input to get its output.
      4. Emit solution lines that a model could reproduce:
            step_1 = <TransformA>(test_input)
            output_grid = <TransformB>(step_1)
    """

    def __init__(self, size=5, min_fill_prob=0.25):
        super().__init__(size)
        self.min_fill_prob = min_fill_prob

        # Registry of (human_name, callable, solution_expr_template)
        # solution_expr_template will be formatted with inp variable name.
        self.transforms = [
            ("mirror_horizontal", MirrorShapeGenerator.mirror_horizontal, "MirrorShapeGenerator.mirror_horizontal({inp})"),
            ("mirror_vertical", MirrorShapeGenerator.mirror_vertical, "MirrorShapeGenerator.mirror_vertical({inp})"),
            ("rotate_clockwise", getattr(RotateShapeGenerator, 'rotate_clockwise', None), "RotateShapeGenerator.rotate_clockwise({inp})"),
            ("rotate_counterclockwise", getattr(RotateShapeGenerator, 'rotate_counterclockwise', None), "RotateShapeGenerator.rotate_counterclockwise({inp})"),
            ("complete_horizontal_symmetry", SymmetryCompleteGenerator.complete_horizontal_symmetry, "SymmetryCompleteGenerator.complete_horizontal_symmetry({inp})"),
            ("complete_vertical_symmetry", SymmetryCompleteGenerator.complete_vertical_symmetry, "SymmetryCompleteGenerator.complete_vertical_symmetry({inp})"),
            ("complete_diagonal_symmetry", SymmetryCompleteGenerator.complete_diagonal_symmetry, "SymmetryCompleteGenerator.complete_diagonal_symmetry({inp})"),
            ("complete_rotational_symmetry", SymmetryCompleteGenerator.complete_rotational_symmetry, "SymmetryCompleteGenerator.complete_rotational_symmetry({inp})"),
            ("count_shapes", CountAndTransformGenerator.count_shapes, "CountAndTransformGenerator.count_shapes({inp})"),
            ("mark_by_size", CountAndTransformGenerator.mark_by_size, "CountAndTransformGenerator.mark_by_size({inp})"),
            ("mark_by_frequency", CountAndTransformGenerator.mark_by_frequency, "CountAndTransformGenerator.mark_by_frequency({inp})"),
            ("count_colors", CountAndTransformGenerator.count_colors, "CountAndTransformGenerator.count_colors({inp})"),
        ]
        # Filter any None callables (in case of missing methods)
        self.transforms = [t for t in self.transforms if t[1] is not None]

    def random_grid(self):
        """Create a random grid with at least one non-zero cell."""
        while True:
            grid = [[0 for _ in range(self.size)] for _ in range(self.size)]
            fill_prob = random.uniform(self.min_fill_prob, 0.45)
            for r in range(self.size):
                for c in range(self.size):
                    if random.random() < fill_prob:
                        grid[r][c] = random.randint(1, 9)
            if any(cell != 0 for row in grid for cell in row):
                return grid

    def apply_transform(self, grid, transform_callable):
        try:
            return transform_callable(grid)
        except Exception:
            # If transform fails, just return original grid (avoid losing example)
            return copy.deepcopy(grid)

    def build_example(self):
        # Pick two transforms (allow same twice)
        t1 = random.choice(self.transforms)
        t2 = random.choice(self.transforms)
        (name1, fn1, expr1) = t1
        (name2, fn2, expr2) = t2

        def build_pair():
            inp = self.random_grid()
            mid = self.apply_transform(inp, fn1)
            out = self.apply_transform(mid, fn2)
            return inp, out

        train_input1, train_output1 = build_pair()
        train_input2, train_output2 = build_pair()
        test_input, test_output = build_pair()

        solution = [
            f"step_1 = {expr1.format(inp='test_input')}",
            f"output_grid = {expr2.format(inp='step_1')}"
        ]

        return {
            'train_input1': train_input1,
            'train_output1': train_output1,
            'train_input2': train_input2,
            'train_output2': train_output2,
            'test_input': test_input,
            'test_output': test_output,
            'solution': solution,
            'transform_chain': [name1, name2]
        }

    def create_fewshot_examples(self, num_examples=100):
        examples = []
        for _ in range(num_examples):
            examples.append(self.build_example())
        return examples
