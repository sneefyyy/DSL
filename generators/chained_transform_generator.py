import random
import copy
from typing import Callable, List, Tuple, Optional

from .base import ExampleGenerator
from .mirror_shape import MirrorShapeGenerator
from .rotate_shape import RotateShapeGenerator
from .symmetry_complete_generator import SymmetryCompleteGenerator
from .flood_fill import FloodFillGenerator
from .move_shape import MoveShapeGenerator
from .count_and_transform_generator import CountAndTransformGenerator
from .create_shape import CreateShapeGenerator
from .connect import ConnectGenerator
from .extract_pattern_generator import ExtractPatternGenerator
from .repeat_pattern_generator import RepeatPatternGenerator


class TransformSpec:
    """Adapter spec that normalizes differently‑shaped transformation functions.

    Each spec provides an apply(...) method returning (new_grid, code_line) where
    code_line is executable Python that, when run in a namespace containing
    the needed classes/functions and with the input variable name, will produce
    the assigned output variable.
    """

    def __init__(self, name: str, apply_fn: Callable):
        self.name = name
        self._apply_fn = apply_fn  # (grid, in_var, out_var, rng) -> (new_grid, code_line)

    def apply(self, grid, in_var: str, out_var: str, rng: random.Random):
        return self._apply_fn(grid, in_var, out_var, rng)


def _mirror_h(grid, in_var, out_var, rng):
    new_grid = MirrorShapeGenerator.mirror_horizontal(grid)
    return new_grid, f"{out_var} = MirrorShapeGenerator.mirror_horizontal({in_var})"


def _mirror_v(grid, in_var, out_var, rng):
    new_grid = MirrorShapeGenerator.mirror_vertical(grid)
    return new_grid, f"{out_var} = MirrorShapeGenerator.mirror_vertical({in_var})"


def _rotate_cw(grid, in_var, out_var, rng):
    if hasattr(RotateShapeGenerator, 'rotate_clockwise'):
        new_grid = RotateShapeGenerator.rotate_clockwise(grid)
        return new_grid, f"{out_var} = RotateShapeGenerator.rotate_clockwise({in_var})"
    return copy.deepcopy(grid), f"{out_var} = {in_var}  # rotate_clockwise missing"


def _rotate_ccw(grid, in_var, out_var, rng):
    if hasattr(RotateShapeGenerator, 'rotate_counterclockwise'):
        new_grid = RotateShapeGenerator.rotate_counterclockwise(grid)
        return new_grid, f"{out_var} = RotateShapeGenerator.rotate_counterclockwise({in_var})"
    return copy.deepcopy(grid), f"{out_var} = {in_var}  # rotate_counterclockwise missing"


def _sym_h(grid, in_var, out_var, rng):
    new_grid = SymmetryCompleteGenerator.complete_horizontal_symmetry(grid)
    return new_grid, f"{out_var} = SymmetryCompleteGenerator.complete_horizontal_symmetry({in_var})"


def _sym_v(grid, in_var, out_var, rng):
    new_grid = SymmetryCompleteGenerator.complete_vertical_symmetry(grid)
    return new_grid, f"{out_var} = SymmetryCompleteGenerator.complete_vertical_symmetry({in_var})"


def _sym_diag(grid, in_var, out_var, rng):
    new_grid = SymmetryCompleteGenerator.complete_diagonal_symmetry(grid)
    return new_grid, f"{out_var} = SymmetryCompleteGenerator.complete_diagonal_symmetry({in_var})"


def _sym_rot(grid, in_var, out_var, rng):
    new_grid = SymmetryCompleteGenerator.complete_rotational_symmetry(grid)
    return new_grid, f"{out_var} = SymmetryCompleteGenerator.complete_rotational_symmetry({in_var})"


def _flood_fill(grid, in_var, out_var, rng):
    h = len(grid)
    w = len(grid[0]) if h else 0
    start = (rng.randrange(h), rng.randrange(w)) if h and w else (0, 0)
    orig = grid[start[0]][start[1]] if h and w else 0
    candidates = [c for c in range(1, 10) if c != orig]
    new_val = rng.choice(candidates) if candidates else ((orig + 1) % 10 or 1)
    new_grid = FloodFillGenerator.flood_fill(grid, start, new_val)
    return new_grid, f"{out_var} = FloodFillGenerator.flood_fill({in_var}, {start}, {new_val})"


def _move_shape(grid, in_var, out_var, rng):
    # choose a center: any non-zero cell or (0,0)
    centers = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v != 0]
    if centers:
        center_row, center_col = rng.choice(centers)
    else:
        center_row = center_col = 0
    delta_row = rng.randint(-2, 2)
    delta_col = rng.randint(-2, 2)
    new_grid = MoveShapeGenerator.move_shape(grid, center_row, center_col, delta_row, delta_col)
    return new_grid, (
        f"{out_var} = MoveShapeGenerator.move_shape({in_var}, {center_row}, {center_col}, {delta_row}, {delta_col})"
    )


def _count_shapes(grid, in_var, out_var, rng):
    new_grid = CountAndTransformGenerator.count_shapes(grid)
    return new_grid, f"{out_var} = CountAndTransformGenerator.count_shapes({in_var})"


def _mark_by_size(grid, in_var, out_var, rng):
    new_grid = CountAndTransformGenerator.mark_by_size(grid)
    return new_grid, f"{out_var} = CountAndTransformGenerator.mark_by_size({in_var})"


def _mark_by_frequency(grid, in_var, out_var, rng):
    new_grid = CountAndTransformGenerator.mark_by_frequency(grid)
    return new_grid, f"{out_var} = CountAndTransformGenerator.mark_by_frequency({in_var})"


def _count_colors(grid, in_var, out_var, rng):
    new_grid = CountAndTransformGenerator.count_colors(grid)
    return new_grid, f"{out_var} = CountAndTransformGenerator.count_colors({in_var})"


# --- Added specs for remaining generators ---
def _create_shape(grid, in_var, out_var, rng):
    # Add a small random shape (2-5 cells) with a random value onto the current grid
    if grid is None:
        # Fallback grid for code emission path (never used for logic)
        grid = [[0]]
    h = len(grid)
    w = len(grid[0]) if h else 0
    k = rng.randint(2, 5)
    points = []
    used = set()
    for _ in range(k):
        if h == 0 or w == 0:
            break
        r, c = rng.randrange(h), rng.randrange(w)
        # Avoid duplicate points
        if (r, c) in used:
            continue
        used.add((r, c))
        points.append((r, c))
    if not points:
        points = [(0, 0)]
    val = rng.randint(1, 9)
    new_grid = CreateShapeGenerator.create_shape(points, val, grid)
    return new_grid, f"{out_var} = CreateShapeGenerator.create_shape({points}, {val}, {in_var})"


def _connect(grid, in_var, out_var, rng):
    if grid is None:
        grid = [[0 for _ in range(5)] for _ in range(5)]
    h = len(grid)
    w = len(grid[0]) if h else 0
    # Ensure start/end satisfy straight or perfect diagonal line
    def sample_point():
        return (rng.randrange(h), rng.randrange(w)) if h and w else (0, 0)
    for _ in range(20):
        start = sample_point()
        end = sample_point()
        if start == end:
            continue
        dr = end[0] - start[0]
        dc = end[1] - start[1]
        if dr == 0 or dc == 0 or abs(dr) == abs(dc):
            break
    else:
        start, end = (0, 0), (0, 0)
    val = rng.randint(1, 9)
    new_grid = ConnectGenerator.connect(grid, start, end, val)
    return new_grid, f"{out_var} = ConnectGenerator.connect({in_var}, {start}, {end}, {val})"


def _extract_most_freq(grid, in_var, out_var, rng):
    new_grid = ExtractPatternGenerator.extract_most_frequent_color(grid)
    return new_grid, f"{out_var} = ExtractPatternGenerator.extract_most_frequent_color({in_var})"


def _extract_largest_shape(grid, in_var, out_var, rng):
    new_grid = ExtractPatternGenerator.extract_largest_shape(grid)
    return new_grid, f"{out_var} = ExtractPatternGenerator.extract_largest_shape({in_var})"


def _extract_non_background(grid, in_var, out_var, rng):
    new_grid = ExtractPatternGenerator.extract_non_background(grid)
    return new_grid, f"{out_var} = ExtractPatternGenerator.extract_non_background({in_var})"


def _repeat_horizontal(grid, in_var, out_var, rng):
    times = rng.randint(2, 4)
    new_grid = RepeatPatternGenerator.repeat_horizontal(grid, times)
    return new_grid, f"{out_var} = RepeatPatternGenerator.repeat_horizontal({in_var}, {times})"


def _repeat_vertical(grid, in_var, out_var, rng):
    times = rng.randint(2, 4)
    new_grid = RepeatPatternGenerator.repeat_vertical(grid, times)
    return new_grid, f"{out_var} = RepeatPatternGenerator.repeat_vertical({in_var}, {times})"


def _repeat_tile(grid, in_var, out_var, rng):
    new_grid = RepeatPatternGenerator.tile_pattern(grid)
    return new_grid, f"{out_var} = RepeatPatternGenerator.tile_pattern({in_var})"


def _repeat_diagonal(grid, in_var, out_var, rng):
    times = rng.randint(3, 5)
    new_grid = RepeatPatternGenerator.repeat_diagonal(grid, times)
    return new_grid, f"{out_var} = RepeatPatternGenerator.repeat_diagonal({in_var}, {times})"


SPECS: List[TransformSpec] = [
    TransformSpec("mirror_horizontal", _mirror_h),
    TransformSpec("mirror_vertical", _mirror_v),
    TransformSpec("rotate_clockwise", _rotate_cw),
    TransformSpec("rotate_counterclockwise", _rotate_ccw),
    TransformSpec("symmetry_horizontal", _sym_h),
    TransformSpec("symmetry_vertical", _sym_v),
    TransformSpec("symmetry_diagonal", _sym_diag),
    TransformSpec("symmetry_rotational", _sym_rot),
    TransformSpec("flood_fill", _flood_fill),
    TransformSpec("move_shape", _move_shape),
    TransformSpec("count_shapes", _count_shapes),
    TransformSpec("mark_by_size", _mark_by_size),
    TransformSpec("mark_by_frequency", _mark_by_frequency),
    TransformSpec("count_colors", _count_colors),
    # Newly added specs
    TransformSpec("create_shape", _create_shape),
    TransformSpec("connect", _connect),
    TransformSpec("extract_most_frequent_color", _extract_most_freq),
    TransformSpec("extract_largest_shape", _extract_largest_shape),
    TransformSpec("extract_non_background", _extract_non_background),
    TransformSpec("repeat_horizontal", _repeat_horizontal),
    TransformSpec("repeat_vertical", _repeat_vertical),
    TransformSpec("repeat_tile", _repeat_tile),
    TransformSpec("repeat_diagonal", _repeat_diagonal),
]


class ChainedTransformGenerator(ExampleGenerator):
    """Generates examples by chaining N randomly chosen adapter-backed transforms.

    Each example includes two training input/output pairs and one test pair.
    The solution lines explicitly list every staged assignment so the model
    can learn multi-step reasoning with parameterized calls.
    """

    def __init__(self, size=5, chain_min=2, chain_max=4, seed: Optional[int] = None):
        super().__init__(size)
        self.chain_min = chain_min
        self.chain_max = chain_max
        self.rng = random.Random(seed)

    def _random_grid(self):
        # ensure at least one non-zero cell
        while True:
            prob = self.rng.uniform(0.20, 0.45)
            g = [[0 for _ in range(self.size)] for _ in range(self.size)]
            for r in range(self.size):
                for c in range(self.size):
                    if self.rng.random() < prob:
                        g[r][c] = self.rng.randint(1, 9)
            if any(v != 0 for row in g for v in row):
                return g

    def _apply_chain(self, start_grid, specs: List[TransformSpec]):
        grid = copy.deepcopy(start_grid)
        # We also capture the sequential intermediate solution lines for test case.
        lines = []
        prev_var = 'test_input'  # overwritten later for actual generation context
        # For generation of output only; lines are produced separately in solution builder.
        for _ in specs:
            # We'll re-simulate with proper variable naming in solution builder.
            pass
        # Actually just produce final grid: apply specs sequentially.
        for spec in specs:
            grid, _ = spec.apply(grid, 'tmp_in', 'tmp_out', self.rng)
        return grid

    def _build_full_solution(self, specs: List[TransformSpec]):
        solution_lines = []
        prev_var = 'test_input'
        for i, spec in enumerate(specs):
            out_var = 'output_grid' if i == len(specs) - 1 else f'step_{i+1}'
            # We need to run the spec again for code emission only (grid not needed here)
            _, code = spec.apply(None if False else [[0]], prev_var, out_var, self.rng)  # dummy grid for code
            # Replace any dummy grid pass-through replacements if earlier added
            # code already references prev_var; safe to append
            solution_lines.append(code)
            prev_var = out_var
        return solution_lines

    def build_example(self):
        chain_length = self.rng.randint(self.chain_min, self.chain_max)
        specs = [self.rng.choice(SPECS) for _ in range(chain_length)]

        def make_pair():
            inp = self._random_grid()
            # Apply chain to produce output
            grid = copy.deepcopy(inp)
            for spec in specs:
                grid, _ = spec.apply(grid, 'x', 'y', self.rng)
            return inp, grid

        train_input1, train_output1 = make_pair()
        train_input2, train_output2 = make_pair()
        test_input, test_output = make_pair()
        # Build solution lines (fresh emission with variable naming). Use a separate RNG clone for deterministic reproduction? We'll accept stochastic emission but deterministic if seed set.
        # Re-emit using a cloned RNG so parameter sampling (for multi-arg) matches test_output transformation sequence.
        # To guarantee reproducibility, we must re-run transforms with SAME sampled params as used on test grid.
        # Simplest: record parameters during test transformation pass.
        solution_lines = []
        prev_var = 'test_input'
        grid_replay = copy.deepcopy(test_input)
        for i, spec in enumerate(specs):
            out_var = 'output_grid' if i == len(specs) - 1 else f'step_{i+1}'
            # Re-run spec on grid_replay, capturing code
            grid_replay, code = spec.apply(grid_replay, prev_var, out_var, self.rng)
            solution_lines.append(code)
            prev_var = out_var
        # grid_replay should equal test_output; if not, fallback (rare if RNG reused differently earlier)
        test_output = grid_replay

        return {
            'train_input1': train_input1,
            'train_output1': train_output1,
            'train_input2': train_input2,
            'train_output2': train_output2,
            'test_input': test_input,
            'test_output': test_output,
            'solution': solution_lines,
            'transform_chain': [s.name for s in specs]
        }

    def create_fewshot_examples(self, num_examples=100):
        examples = []
        for _ in range(num_examples):
            examples.append(self.build_example())
        return examples
