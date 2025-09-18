"""Standalone verification of all chained pipeline gold solutions.

Loads JSON_training/ChainedPipeline.json and re-executes every solution block
in a clean interpreter namespace mirroring dataset generation environment.
Reports overall pass rate plus per chain length stats and first N failures
(with diffs).
"""
from __future__ import annotations
import json
import copy
from pathlib import Path
from typing import Dict, Any, List, Tuple
import math
import argparse

# Reuse the existing interpreter utilities
import sys
# Ensure project root on path so `Interpreter` module resolves when running as script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Interpreter import ExampleInterpreter

DATASET_PATH = Path('JSON_training/ChainedPipeline.json')


def wilson_interval(success: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p_hat = success / total
    denom = 1 + z**2 / total
    center = p_hat + z**2 / (2 * total)
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total)
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return lower, upper


def diff_grids(a: List[List[int]], b: List[List[int]]) -> List[str]:
    out = []
    if a is None or b is None:
        return ["<None grid>"]
    h = max(len(a), len(b))
    for r in range(h):
        row_a = a[r] if r < len(a) else []
        row_b = b[r] if r < len(b) else []
        w = max(len(row_a), len(row_b))
        parts = []
        for c in range(w):
            va = row_a[c] if c < len(row_a) else None
            vb = row_b[c] if c < len(row_b) else None
            if va == vb:
                parts.append(str(va) if va not in (0, None) else '.')
            else:
                parts.append(f"{va}->{vb}")
        out.append(' '.join(parts))
    return out


def verify_all(limit_fail_details: int = 20, verbose_fail: bool = True):
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")

    data = json.loads(DATASET_PATH.read_text())
    interp = ExampleInterpreter()

    total = len(data)
    passed = 0
    failures: List[Dict[str, Any]] = []
    per_length = {}
    per_length_pass = {}

    for idx, ex in enumerate(data):
        length = len(ex.get('transform_chain', ex.get('solution', [])))
        per_length[length] = per_length.get(length, 0) + 1

        success, produced = interp.test_example(ex)
        if success:
            passed += 1
            per_length_pass[length] = per_length_pass.get(length, 0) + 1
        else:
            failures.append({
                'index': idx,
                'chain_length': length,
                'transform_chain': ex.get('transform_chain'),
                'solution': ex['solution'],
                'test_input': ex['test_input'],
                'expected': ex['test_output'],
                'produced': produced
            })

    print("\n=== Chained Gold Solution Verification ===")
    print(f"Total examples: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    acc = passed / total if total else 0.0
    lo, hi = wilson_interval(passed, total)
    print(f"Accuracy: {acc*100:.2f}%  (Wilson 95% CI: [{lo*100:.2f}%, {hi*100:.2f}%])")

    print("\nPer chain length:")
    for L in sorted(per_length):
        tot_L = per_length[L]
        pass_L = per_length_pass.get(L, 0)
        loL, hiL = wilson_interval(pass_L, tot_L)
        print(f"  L={L}: {pass_L}/{tot_L} = {pass_L/tot_L*100:.2f}%  CI[{loL*100:.2f}%, {hiL*100:.2f}%]")

    if failures and verbose_fail:
        print(f"\nFirst {min(limit_fail_details, len(failures))} failure details:")
        for f in failures[:limit_fail_details]:
            print("-" * 60)
            print(f"Index: {f['index']}  ChainLen: {f['chain_length']}  Chain: {f['transform_chain']}")
            print("Solution lines:")
            for line in f['solution']:
                print(f"  {line}")
            print("Diff (expected vs produced):")
            if f['produced'] is None:
                print("  <No produced output_grid>")
            else:
                diff = diff_grids(f['expected'], f['produced'])
                for dline in diff:
                    print("  "+dline)

    return passed == total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-fails', action='store_true', help='Suppress failure details output')
    parser.add_argument('--fail-limit', type=int, default=20)
    args = parser.parse_args()
    verify_all(limit_fail_details=args.fail_limit, verbose_fail=not args.no_fails)


if __name__ == '__main__':
    main()
