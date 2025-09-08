import json, random, argparse, sys
from pathlib import Path
import copy

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dsl import dsl_functions as DF


def random_points(n=4, size=5):
    pts = set()
    while len(pts) < n:
        pts.add((random.randrange(size), random.randrange(size)))
    return list(pts)


def build_example(func_name, size=5):
    # Create base train/test inputs
    grid1 = [[0]*size for _ in range(size)]
    grid2 = [[0]*size for _ in range(size)]
    test_grid = [[0]*size for _ in range(size)]
    # sprinkle random points
    for g in (grid1, grid2, test_grid):
        for _ in range(random.randint(4,8)):
            r = random.randrange(size); c = random.randrange(size)
            g[r][c] = random.randint(1,9)

    # Parameter sampling per function
    if func_name in {'connect'}:
        p1 = (0,0); p2 = (size-1, size-1)
        out1 = DF.connect(p1,p2)
        out2 = DF.connect((0,size-1),(size-1,0))
        test_out = DF.connect((0, size//2),(size-1,size//2))
        solution = [f"output_grid = connect({(0, size//2)}, {(size-1,size//2)})"]
        train_out1, train_out2 = out1, out2
        grid1_in, grid2_in, test_in = out1, out2, test_out  # treat connect outputs as base (no separate inputs)
        test_out_final = test_out
    elif func_name == 'flood_fill':
        start = (0,0)
        new_val = 9
        train_out1 = DF.flood_fill(grid1, start, new_val)
        train_out2 = DF.flood_fill(grid2, start, new_val)
        test_out_final = DF.flood_fill(test_grid, start, new_val)
        solution = [f"output_grid = flood_fill(test_input, {start}, {new_val})"]
        grid1_in, grid2_in, test_in = grid1, grid2, test_grid
    elif func_name == 'mirror_shape':
        axis = random.choice(['horizontal','vertical'])
        train_out1 = DF.mirror_shape(random_points(5,size), axis)
        train_out2 = DF.mirror_shape(random_points(5,size), axis)
        test_pts = random_points(5,size)
        test_out_final = DF.mirror_shape(test_pts, axis)
        solution = [f"output_grid = mirror_shape({test_pts}, '{axis}')"]
        grid1_in, grid2_in, test_in = train_out1, train_out2, test_out_final
    elif func_name == 'complete_symmetry':
        axis = random.choice(['horizontal','vertical','diagonal'])
        train_out1 = DF.complete_symmetry(grid1, axis)
        train_out2 = DF.complete_symmetry(grid2, axis)
        test_out_final = DF.complete_symmetry(test_grid, axis)
        solution = [f"output_grid = complete_symmetry(test_input, '{axis}')"]
        grid1_in, grid2_in, test_in = grid1, grid2, test_grid
    elif func_name == 'count_shapes':
        train_out1 = DF.count_shapes(grid1)
        train_out2 = DF.count_shapes(grid2)
        test_out_final = DF.count_shapes(test_grid)
        solution = ["output_grid = count_shapes(test_input)"]
        grid1_in, grid2_in, test_in = grid1, grid2, test_grid
    elif func_name == 'mark_by_size':
        train_out1 = DF.mark_by_size(grid1)
        train_out2 = DF.mark_by_size(grid2)
        test_out_final = DF.mark_by_size(test_grid)
        solution = ["output_grid = mark_by_size(test_input)"]
        grid1_in, grid2_in, test_in = grid1, grid2, test_grid
    elif func_name == 'mark_by_frequency':
        train_out1 = DF.mark_by_frequency(grid1)
        train_out2 = DF.mark_by_frequency(grid2)
        test_out_final = DF.mark_by_frequency(test_grid)
        solution = ["output_grid = mark_by_frequency(test_input)"]
        grid1_in, grid2_in, test_in = grid1, grid2, test_grid
    elif func_name == 'count_colors':
        train_out1 = DF.count_colors(grid1)
        train_out2 = DF.count_colors(grid2)
        test_out_final = DF.count_colors(test_grid)
        solution = ["output_grid = count_colors(test_input)"]
        grid1_in, grid2_in, test_in = grid1, grid2, test_grid
    else:  # generic fallback (identity)
        train_out1 = copy.deepcopy(grid1)
        train_out2 = copy.deepcopy(grid2)
        test_out_final = copy.deepcopy(test_grid)
        solution = ["output_grid = test_input  # identity placeholder"]
        grid1_in, grid2_in, test_in = grid1, grid2, test_grid

    return {
        'train_input1': grid1_in,
        'train_output1': train_out1,
        'train_input2': grid2_in,
        'train_output2': train_out2,
        'test_input': test_in,
        'test_output': test_out_final,
        'solution': solution,
        'dsl_functions_used': [func_name]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='JSON_training/DSLFunctionTest.json')
    ap.add_argument('--per-func', type=int, default=5)
    args = ap.parse_args()

    random.seed(123)
    examples = []
    for fn in DF.ALL_DSL_FUNCTIONS:
        for _ in range(args.per_func):
            examples.append(build_example(fn))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out,'w') as f:
        json.dump(examples, f, indent=2)
    print(f"Wrote {len(examples)} DSL function test examples to {args.out}")


if __name__ == '__main__':
    main()
