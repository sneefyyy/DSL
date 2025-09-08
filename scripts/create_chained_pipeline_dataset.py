import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generators.chained_transform_generator import ChainedTransformGenerator


def validate_examples(examples, max_check=50):
    import copy
    # Build execution namespace similar to interpreter
    from generators import (
        MirrorShapeGenerator,
        RotateShapeGenerator,
        SymmetryCompleteGenerator,
        FloodFillGenerator,
        MoveShapeGenerator,
        CountAndTransformGenerator,
        CreateShapeGenerator,
        ConnectGenerator,
        ExtractPatternGenerator,
        RepeatPatternGenerator,
    )
    ns_base = {
        'MirrorShapeGenerator': MirrorShapeGenerator,
        'RotateShapeGenerator': RotateShapeGenerator,
        'SymmetryCompleteGenerator': SymmetryCompleteGenerator,
        'FloodFillGenerator': FloodFillGenerator,
        'MoveShapeGenerator': MoveShapeGenerator,
        'CountAndTransformGenerator': CountAndTransformGenerator,
        'CreateShapeGenerator': CreateShapeGenerator,
        'ConnectGenerator': ConnectGenerator,
        'ExtractPatternGenerator': ExtractPatternGenerator,
        'RepeatPatternGenerator': RepeatPatternGenerator,
    }
    ok = 0
    for ex in examples[:max_check]:
        ns = ns_base.copy()
        ns['test_input'] = copy.deepcopy(ex['test_input'])
        try:
            exec('\n'.join(ex['solution']), ns)
            if ns.get('output_grid') == ex['test_output']:
                ok += 1
        except Exception:
            pass
    return ok, min(len(examples), max_check)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300, help='Total examples (will be adjusted to satisfy min per length if needed)')
    ap.add_argument('--size', type=int, default=5)
    ap.add_argument('--out', type=str, default='JSON_training/ChainedPipeline.json')
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--min-per-length', type=int, default=75, help='Minimum examples for each chain length (2,3,4)')
    args = ap.parse_args()

    # We will explicitly generate balanced sets for lengths 2,3,4.
    target_lengths = [2,3,4]
    need_each = args.min_per_length
    total_needed = need_each * len(target_lengths)
    if args.n < total_needed:
        args.n = total_needed

    remaining = args.n
    gen = ChainedTransformGenerator(size=args.size, seed=args.seed, chain_min=2, chain_max=4)
    examples = []
    # Deterministic RNG already seeded inside generator; we override chain bounds manually per slice.
    for chain_len in target_lengths:
        count = need_each
        for _ in range(count):
            ex = gen.build_example()
            ex['chain_length'] = len(ex['transform_chain'])
            while ex['chain_length'] != chain_len:
                # rebuild until desired length
                ex = gen.build_example()
                ex['chain_length'] = len(ex['transform_chain'])
            examples.append(ex)
        remaining -= count
    # Fill any remaining quota with random lengths (2-4)
    while len(examples) < args.n:
        ex = gen.build_example()
        ex['chain_length'] = len(ex['transform_chain'])
        examples.append(ex)

    ok, total = validate_examples(examples)
    print(f"Validation sample: {ok}/{total} ({ok/total*100:.1f}%)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as f:
        json.dump(examples, f, indent=2)
    # Report distribution
    from collections import Counter
    dist = Counter(ex['chain_length'] for ex in examples)
    print('Length distribution:', dict(dist))
    print(f"Wrote {len(examples)} chained pipeline examples to {out_path}")


if __name__ == '__main__':
    main()
