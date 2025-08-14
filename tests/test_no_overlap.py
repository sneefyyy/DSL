#!/usr/bin/env python3
"""
Check that no example appears in more than one split of the saved dataset.

This script loads the dataset saved to `dsl_dataset/` (created by
`create_hf_dataset.py`) and asserts there is no overlap between the
`train`, `validation`, and `test` splits. It exits with code 0 on success
and non-zero on failure.
"""
import json
import sys
from datasets import load_from_disk


def canonicalize(example):
    """Return a stable string representation for an example.

    Uses JSON with sorted keys when possible, falls back to str().
    """
    try:
        return json.dumps(example, sort_keys=True, default=str)
    except Exception:
        return str(example)


def main(dataset_dir="dsl_dataset"):
    ds = load_from_disk(dataset_dir)

    splits = [s for s in ("train", "validation", "test")]

    seen = {}  # map from canonicalized example -> (split, index)
    overlaps = []

    for split in splits:
        if split not in ds:
            print(f"Warning: split '{split}' not found in {dataset_dir}; skipping")
            continue
        for idx, ex in enumerate(ds[split]):
            key = canonicalize(ex)
            if key in seen:
                overlaps.append((seen[key], (split, idx)))
            else:
                seen[key] = (split, idx)

    if overlaps:
        print("FAIL: Found overlapping examples across splits:")
        for (s1, i1), (s2, i2) in overlaps:
            print(f" - {s1}[{i1}] == {s2}[{i2}]")
        return 2

    print("PASS: No examples overlap across splits.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
