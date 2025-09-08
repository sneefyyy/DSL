"""Canonical DSL function wrappers for testing.

These provide simplified, uniform interfaces that map onto existing generator
implementations. Designed ONLY for test dataset generation / evaluation, not
for model training semantics fidelity.
"""
from typing import List, Tuple
import copy
import random

from generators.mirror_shape import MirrorShapeGenerator
from generators.rotate_shape import RotateShapeGenerator
from generators.symmetry_complete_generator import SymmetryCompleteGenerator
from generators.flood_fill import FloodFillGenerator
from generators.move_shape import MoveShapeGenerator
from generators.count_and_transform_generator import CountAndTransformGenerator
from generators.create_shape import CreateShapeGenerator

Point = Tuple[int, int]


def _blank_like(grid):
    return [[0 for _ in row] for row in grid]


def connect(point_1: Point, point_2: Point):  # returns a new grid with a line between points
    (r1, c1), (r2, c2) = point_1, point_2
    size = max(r1, r2, c1, c2) + 1
    grid = [[0 for _ in range(size)] for _ in range(size)]
    dr = 0 if r1 == r2 else (1 if r2 > r1 else -1)
    dc = 0 if c1 == c2 else (1 if c2 > c1 else -1)
    r, c = r1, c1
    grid[r][c] = 1
    while (r, c) != (r2, c2):
        if r != r2:
            r += dr
        if c != c2:
            c += dc
        grid[r][c] = 1
    return grid


def move_shape(points: List[Point], direction: str, amount: int, grid=None):
    if grid is None:
        size = max(max(r for r, _ in points)+1, max(c for _, c in points)+1, 5)
        grid = [[0]*size for _ in range(size)]
        for r,c in points:
            grid[r][c] = 1
    dr, dc = 0,0
    if direction == 'up': dr = -amount
    elif direction == 'down': dr = amount
    elif direction == 'left': dc = -amount
    elif direction == 'right': dc = amount
    # find a center (first non-zero)
    centers = [(r,c) for r,row in enumerate(grid) for c,v in enumerate(row) if v!=0]
    if centers:
        cr, cc = centers[0]
        return MoveShapeGenerator.move_shape(grid, cr, cc, dr, dc)
    return grid


def rotate_shape(points: List[Point], center_point: Point, rotations: int, direction: str, grid=None):
    if grid is None:
        size = max(max(r for r,_ in points)+1, max(c for _,c in points)+1, 5)
        grid = [[0]*size for _ in range(size)]
        for r,c in points:
            grid[r][c] = 1
    # collapse to using rotate function on points
    times = rotations % 4
    if times == 0:
        return grid
    dir_flag = 'clockwise' if direction.lower().startswith('c') else 'counterclockwise'
    rotated = RotateShapeGenerator.rotate(points, center_point, times, dir_flag)
    out = _blank_like(grid)
    for r,c in rotated:
        if 0 <= r < len(out) and 0 <= c < len(out):
            out[r][c] = 1
    return out


def create_shape(points: List[Point], value: int, grid=None):
    return CreateShapeGenerator.create_shape(points, value, grid)


def flood_fill(grid, start_point: Point, new_value: int):
    return FloodFillGenerator.flood_fill(grid, start_point, new_value)


def mirror_shape(points: List[Point], axis: str, grid=None):
    if grid is None:
        size = max(max(r for r,_ in points)+1, max(c for _,c in points)+1, 5)
        grid = [[0]*size for _ in range(size)]
        for r,c in points:
            grid[r][c] = 1
    if axis == 'horizontal':
        return MirrorShapeGenerator.mirror_horizontal(grid)
    return MirrorShapeGenerator.mirror_vertical(grid)


def complete_symmetry(grid, axis: str):
    if axis == 'horizontal':
        return SymmetryCompleteGenerator.complete_horizontal_symmetry(grid)
    if axis == 'vertical':
        return SymmetryCompleteGenerator.complete_vertical_symmetry(grid)
    # fallback: diagonal (reuse rotational if axis == 'rotational')
    if axis == 'diagonal':
        return SymmetryCompleteGenerator.complete_diagonal_symmetry(grid)
    return SymmetryCompleteGenerator.complete_rotational_symmetry(grid)


def extract_pattern(grid, pattern_type: str):  # placeholder (identity)
    return copy.deepcopy(grid)


def repeat_pattern(pattern_grid, times: int, direction: str):  # placeholder replicate horizontally/vertically
    base = pattern_grid
    if direction == 'horizontal':
        return [row * times for row in base]
    # vertical
    out = []
    for _ in range(times):
        out += copy.deepcopy(base)
    return out


def count_shapes(grid):
    return CountAndTransformGenerator.count_shapes(grid)


def mark_by_size(grid):
    return CountAndTransformGenerator.mark_by_size(grid)


def mark_by_frequency(grid):
    return CountAndTransformGenerator.mark_by_frequency(grid)


def count_colors(grid):
    return CountAndTransformGenerator.count_colors(grid)


ALL_DSL_FUNCTIONS = [
    'connect','move_shape','rotate_shape','create_shape','flood_fill','mirror_shape',
    'complete_symmetry','extract_pattern','repeat_pattern','count_shapes','mark_by_size',
    'mark_by_frequency','count_colors'
]
