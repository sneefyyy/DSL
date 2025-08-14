import json
from pathlib import Path
import importlib.util

# Load create_hf_dataset.py directly from repository root so tests don't depend
# on sys.path or package layout.
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "create_hf_dataset", str(root / "create_hf_dataset.py")
)
create_hf_dataset_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(create_hf_dataset_mod)
create_hf_dataset = create_hf_dataset_mod.create_hf_dataset


def canonicalize(example):
    try:
        return json.dumps(example, sort_keys=True, default=str)
    except Exception:
        return str(example)


def test_no_overlap():
    # Build the dataset in-memory from the JSON_training/ directory so tests
    # do not rely on a local saved dsl_dataset/ directory.
    ds = create_hf_dataset('JSON_training/', train_frac=0.90, val_frac=0.05, seed=42)

    splits = ["train", "validation", "test"]

    seen = {}
    overlaps = []

    for split in splits:
        if split not in ds:
            raise AssertionError(f"Expected split '{split}' but it was not found")
        for idx, ex in enumerate(ds[split]):
            key = canonicalize(ex)
            if key in seen:
                overlaps.append((seen[key], (split, idx)))
            else:
                seen[key] = (split, idx)

    assert not overlaps, f"Found overlapping examples across splits: {overlaps}"
