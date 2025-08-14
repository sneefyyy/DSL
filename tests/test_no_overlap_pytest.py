import json
from datasets import load_from_disk


def canonicalize(example):
    try:
        return json.dumps(example, sort_keys=True, default=str)
    except Exception:
        return str(example)


def test_no_overlap():
    ds = load_from_disk("dsl_dataset")

    splits = ["train", "validation", "test"]

    seen = {}
    overlaps = []

    for split in splits:
        if split not in ds:
            # If a split is missing, test should fail
            raise AssertionError(f"Expected split '{split}' in dsl_dataset but it was not found")
        for idx, ex in enumerate(ds[split]):
            key = canonicalize(ex)
            if key in seen:
                overlaps.append((seen[key], (split, idx)))
            else:
                seen[key] = (split, idx)

    assert not overlaps, f"Found overlapping examples across splits: {overlaps}"
